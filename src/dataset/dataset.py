import random
from copy import deepcopy

import torch
from torch.utils.data import Dataset

from src.utils.ogm_util import create_OGM_ego
from src.utils.tracks_import import read_from_csv

import json
import sys
import cv2
import numpy as np
from pathlib import Path
import re
import os

from loguru import logger
import pandas as pd
import pickle


class OGMDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, config):

        scene_ids = set()
        for fname in os.listdir(config['dataset_dir']):
            match = re.match(r"(\d+)_.*\.csv", fname)
            if match:
                scene_ids.add(match.group(1))

        scene_ids = sorted(scene_ids)
        print(f"Found scenes: {scene_ids}")

        self.input_path = config['dataset_dir']

        self.tracks = []
        self.tracks_meta = []
        self.visibility_data = []
        self.background_images = {}
        self.fixed_blocks_info = {}
        self.frame_to_track_idxs = {}

        for scene_id in scene_ids:

            if int(scene_id) < 11:
                continue

            tracks_file = os.path.join(self.input_path, f"{scene_id}_tracks.csv")
            tracks_meta_file = os.path.join(self.input_path, f"{scene_id}_tracksMeta.csv")
            recording_meta_file = os.path.join(self.input_path, f"{scene_id}_recordingMeta.csv")
            fixed_blocks_file = os.path.join(self.input_path, f"{scene_id}_fixedBlocks.csv")
            visibility_file = os.path.join(self.input_path, f"{scene_id}_visibilityData.csv")

            tracks, tracks_meta, recording_meta, fixed_blocks_info = read_from_csv(
                tracks_file, tracks_meta_file, recording_meta_file, fixed_blocks_file, include_px_coordinates=True
            )

            visibility_df = pd.read_csv(visibility_file)
            self.fixed_blocks_info[int(scene_id)] = fixed_blocks_info

            # Collect DataFrames
            self.tracks.append(pd.DataFrame(tracks))
            self.tracks_meta.append(pd.DataFrame(tracks_meta))
            self.visibility_data.append(visibility_df)

            # Store background images for scenes
            bg_path = os.path.join(self.input_path, 'semantic_maps', f"{scene_id}_background.png")
            img = cv2.imread(bg_path)
            self.background_images[int(scene_id)] = img

        self.config = config
        self.input_path = config["dataset_dir"]
        self.dataset = config["dataset"].lower()
        self.history_length = config["history_length"]
        self.num_features = 7  # x, y, heading, xVelocity, yVelocity, xAcceleration, yAcceleration

        # Load dataset specific visualization parameters from file
        dataset_params_path = Path(config["visualizer_params_dir"]) / "visualizer_params.json"

        if not dataset_params_path.exists():
            logger.error("Could not find dataset visualization parameters in {}", dataset_params_path)
            sys.exit(-1)

        with open(dataset_params_path) as f:
            self.dataset_params = json.load(f)

        if self.dataset not in self.dataset_params["datasets"]:
            logger.error("Visualization parameters for dataset {} not found in {}. Please make sure, that the needed "
                         "parameters are given", self.dataset, dataset_params_path)
            sys.exit(-1)

        self.dataset_params = self.dataset_params["datasets"][self.dataset]
        self.scale_down_factor = self.dataset_params["scale_down_factor"]

        self.tracks = pd.concat(self.tracks, ignore_index=True)
        self.tracks_meta = pd.concat(self.tracks_meta, ignore_index=True)
        self.visibility_data = pd.concat(self.visibility_data, ignore_index=True)

        self.data_dict = {}
        # Check index file saved into a file
        index_file_path = Path(os.path.join(self.input_path, "index_map2.pkl"))
        if index_file_path.exists():
            logger.info("Loading index map from {}", index_file_path)
            self.data_dict = pickle.load(open(index_file_path, "rb"))
        else:
            for scene_id in scene_ids:
                scene_id = int(scene_id)
                if scene_id < 11:
                    continue

                tracks_meta = self.tracks_meta[(self.tracks_meta["recordingId"] == scene_id)]
                visibility_data = self.visibility_data[(self.visibility_data["recordingId"] == scene_id)]
                tracks = self.tracks[(self.tracks["recordingId"] == scene_id)]

                # Determine the first and last frame
                minimum_frame = tracks_meta["initialFrame"].min()
                maximum_frame = tracks_meta["finalFrame"].max()

                # Create a mapping between frame and idxs of tracks for quick lookup during playback
                frame_to_track_idxs = {}
                for i_frame in range(minimum_frame, maximum_frame + 1):
                    indices = \
                    tracks_meta[(tracks_meta["initialFrame"] <= i_frame) & (tracks_meta["finalFrame"] >= i_frame)][
                        "trackId"].tolist()
                    frame_to_track_idxs[i_frame] = indices

                # We have to find out timesteps that have atleast one hidden record in the visiblity data.
                # We cannot start from the minimum_frame as we have to include the history as well.
                for i_frame in range(minimum_frame + self.history_length, maximum_frame + 1,
                                     (self.history_length * 2)):

                    hidden_objects = visibility_data[(visibility_data['frame'] == i_frame) &
                                                     (visibility_data['visibility'] == False) &
                                                     (visibility_data['located'] == 'FRONT')]

                    visible_objects = visibility_data[(visibility_data['frame'] == i_frame) &
                                                      (visibility_data['visibility'] == True) &
                                                      (visibility_data['located'] == 'FRONT')]

                    if len(hidden_objects) == 0 or len(visible_objects) == 0: continue

                    tracks_with_full_history = []
                    tracks_with_velocity = []
                    for track_idx in frame_to_track_idxs[i_frame]:
                        track_meta_i = tracks_meta[tracks_meta["trackId"] == track_idx]
                        initial_frame = track_meta_i["initialFrame"].item()
                        vehicle_type = track_meta_i["class"].item()
                        # Check if the track has a full history
                        if initial_frame <= (i_frame - self.history_length) and (vehicle_type == 'car'):
                            tracks_with_full_history.append(track_idx)

                        track = tracks[(tracks["trackId"] == track_idx)]
                        current_index = self._get_current_index(track_meta_i, i_frame)
                        x_velocity = track["xVelocity"].values[0][current_index]
                        y_velocity = track["yVelocity"].values[0][current_index]

                        # Check if the track has a non-zero velocity
                        if x_velocity != 0 or y_velocity != 0:
                            tracks_with_velocity.append(track_idx)

                    tracks_with_limited_visibility = list(hidden_objects["trackId"].drop_duplicates())
                    tracks_with_front_visible_objects = list(visible_objects["trackId"].drop_duplicates())

                    # Find intersection of tracks with limited visibility, tracks with full history and front visible objects
                    eligible_tracks = list(set(tracks_with_limited_visibility) &
                                           set(tracks_with_full_history) &
                                           set(tracks_with_front_visible_objects) &
                                           set(tracks_with_velocity))
                    if len(eligible_tracks) == 0: continue

                    ego_vehicle_track_idx = random.choice(eligible_tracks)
                    # extract historical data for the ego vehicle
                    historical_adjacent_obs, historical_ego_obs, map_obs = self._extract_historical_data(
                        ego_vehicle_track_idx, scene_id, i_frame)

                    # extract ground truth data for the ego vehicle
                    visible_tracks_pts, hidden_tracks_pts = self._extract_ground_truth_data(
                        ego_vehicle_track_idx, scene_id, i_frame)

                    self.data_dict[str(scene_id) + '_' + str(i_frame) + '_' + str(ego_vehicle_track_idx)] = {
                        "historical_adjacent_obs": historical_adjacent_obs,
                        "historical_ego_obs": historical_ego_obs,
                        "map_obs": None,
                        "visible_tracks_pts": visible_tracks_pts,
                        "hidden_tracks_pts": hidden_tracks_pts
                    }

            pickle.dump(self.data_dict, open(index_file_path, "wb"))

        print("Done Loading")

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        key = list(self.data_dict.keys())[idx]
        keys = key.split("_")
        scene_id = int(keys[0])
        current_frame = int(keys[1])
        ego_vehicle_track_idx = int(keys[2])

        data_dict = self.data_dict[key]

        # map
        backgrond_img = self.background_images[scene_id]
        gt_background_img = deepcopy(backgrond_img)
        resized_map = cv2.resize(gt_background_img, (224, 224), interpolation=cv2.INTER_AREA)

        # Historical observations
        historical_adjacent_obs, historical_ego_obs, map_obs, visible_tracks_pts, hidden_tracks_pts = (
            data_dict["historical_adjacent_obs"], data_dict["historical_ego_obs"], data_dict["map_obs"],
            data_dict["visible_tracks_pts"], data_dict["hidden_tracks_pts"])

        # Extract Ground Truth
        ego_track = self.tracks[(self.tracks["trackId"] == ego_vehicle_track_idx)
                                & (self.tracks["recordingId"] == scene_id)].iloc[0].to_dict()
        ego_track_meta = self.tracks_meta[(self.tracks_meta["trackId"] == ego_vehicle_track_idx)
                                          & (self.tracks_meta["recordingId"] == scene_id)].iloc[0].to_dict()
        pts_ego = self._extract_track_info(ego_track, ego_track_meta, current_frame, backgrond_img.shape)["pts"]
        heading_ego = self._get_heading(ego_track, ego_track_meta, current_frame)

        # Create OGM
        ogm, ogm_gt = create_OGM_ego(pts_ego.squeeze(), heading_ego, visible_tracks_pts, hidden_tracks_pts,
                                     gt_background_img, self.fixed_blocks_info[scene_id])

        return (historical_adjacent_obs, historical_ego_obs, map_obs, ogm, ogm_gt, resized_map)


    def __getitem__2(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        key = list(self.data_dict.keys())[idx]
        scene_id = int(key.split("_")[0])
        current_frame = int(key.split("_")[1])

        ego_vehicle_track_idx = self.data_dict[key]
        ego_track = self.tracks[(self.tracks["trackId"] == ego_vehicle_track_idx)
                                & (self.tracks["recordingId"] == scene_id)].iloc[0].to_dict()
        ego_track_meta = self.tracks_meta[(self.tracks_meta["trackId"] == ego_vehicle_track_idx)
                                          & (self.tracks_meta["recordingId"] == scene_id)].iloc[0].to_dict()

        # iterate through all the historical frames upto the current_frame
        historical_adjacent_obs, historical_ego_obs, map_obs = {}, [], []

        backgrond_img = self.background_images[scene_id]
        starting_frame = current_frame - self.history_length
        for t in range(starting_frame, current_frame):  # T, N, D
            map = deepcopy(backgrond_img)
            visible_track_ids = list(self.visibility_data[(self.visibility_data["trackId"] == ego_vehicle_track_idx)
                                                          & (self.visibility_data["frame"] == t)
                                                          & (self.visibility_data["visibility"] == True)
                                                          & (self.visibility_data["recordingId"] == scene_id)
                                                          & (self.visibility_data["located"] == 'FRONT')]
                                     ["adjacentTrackId"].drop_duplicates())

            # Extract historical observations for visible tracks
            recorded_tack_ids = []
            for track_idx in visible_track_ids:
                track = self.tracks[(self.tracks["trackId"] == track_idx)
                                    & (self.tracks["recordingId"] == scene_id)].iloc[0].to_dict()
                track_meta = self.tracks_meta[(self.tracks_meta["trackId"] == track_idx)
                                              & (self.tracks_meta["recordingId"] == scene_id)].iloc[0].to_dict()

                track_info = self._extract_track_info(track, track_meta, t, backgrond_img.shape)
                cv2.fillPoly(map, [track_info["pts"]], (150, 150, 50))  #150, 100, 150

                track_data = np.array([track_info["center"][0], track_info["center"][1], track_info["heading"],
                                       track_info["xVelocity"], track_info["yVelocity"], track_info["xAcceleration"],
                                       track_info["yAcceleration"]])

                if track_idx not in historical_adjacent_obs.keys():
                    if t > starting_frame:
                        # This object appeared lately. So have to add nulls/empty/zeros for previous frames
                        historical_adjacent_obs[track_idx] = [[0] * self.num_features for _ in range(starting_frame, t)]
                        historical_adjacent_obs[track_idx].append(track_data)
                    else:
                        historical_adjacent_obs[track_idx] = [track_data]
                else:
                    historical_adjacent_obs[track_idx].append(track_data)

                recorded_tack_ids.append(track_idx)

            # Fill the historical observations with zeros for the tracks that are not visible in the current frame
            for track_idx in historical_adjacent_obs.keys():
                if track_idx not in recorded_tack_ids:
                    historical_adjacent_obs[track_idx].append([0])

            # Extract historical observations for the ego vehicle
            track_info = self._extract_track_info(ego_track, ego_track_meta, t, backgrond_img.shape)
            cv2.fillPoly(map, [track_info["pts"]], (255, 180, 200))
            historical_ego_obs.append(np.array([track_info["center"][0],
                                                track_info["center"][1],
                                                track_info["heading"],
                                                track_info["xVelocity"],
                                                track_info["yVelocity"],
                                                track_info["xAcceleration"],
                                                track_info["yAcceleration"]]))

            map_resized = cv2.resize(map, (224, 224), interpolation=cv2.INTER_AREA)
            map_obs.append(map_resized)

        # cv2.imwrite("history.png", map)

        # Extract Ground Truth
        pts_ego = self._extract_track_info(ego_track, ego_track_meta, current_frame, backgrond_img.shape)["pts"]
        heading_ego = self._get_heading(ego_track, ego_track_meta, current_frame)

        gt_background_img = deepcopy(backgrond_img)

        # Create OGM
        all_tracks_visibility_df = self.visibility_data[(self.visibility_data["frame"] == current_frame)
                                                      & (self.visibility_data["trackId"] == ego_vehicle_track_idx)
                                                      & (self.visibility_data["recordingId"] == scene_id)
                                                      & (self.visibility_data["located"] == 'FRONT')]

        all_track_idx = all_tracks_visibility_df["adjacentTrackId"].drop_duplicates().tolist()
        visible_track_idx = (all_tracks_visibility_df[all_tracks_visibility_df["visibility"] == True]["adjacentTrackId"]
                             .drop_duplicates().tolist())
        all_tracks = self.tracks[(self.tracks["trackId"].isin(all_track_idx))
                                     & (self.tracks["recordingId"] == scene_id)].to_dict('records')

        visible_pts = []
        hidden_tracks_pts = []
        for track in all_tracks:
            track_meta = self.tracks_meta[(self.tracks_meta["trackId"] == track["trackId"])
                                          & (self.tracks_meta["recordingId"] == scene_id)].iloc[0].to_dict()
            pts = self._extract_track_info(track, track_meta, current_frame, backgrond_img.shape)["pts"].squeeze()

            if track["trackId"] in visible_track_idx:
                visible_pts.append(pts)
            else:
                hidden_tracks_pts.append(pts)

            # cv2.fillPoly(backgrond_img, [pts], (0, 255, 255))

        # cv2.imshow('Image with Polygon', backgrond_img)

        ogm, ogm_gt = create_OGM_ego(pts_ego.squeeze(), heading_ego, visible_pts, hidden_tracks_pts, gt_background_img,
                                     self.fixed_blocks_info[scene_id])

        return (historical_adjacent_obs, historical_ego_obs, map_obs, ogm, ogm_gt)

    def _extract_track_info(self, track, track_meta, t: int, width_height: tuple) -> dict:
        current_index = self._get_current_index(track_meta, t)
        if current_index < 0:
            return {}

        object_class = track_meta["class"]
        if track["bboxVis"] is not None:
            bounding_box = track["bboxVis"][current_index] / self.scale_down_factor
        else:
            bounding_box = None
        center_points = track["centerVis"] / self.scale_down_factor
        center_point = center_points[current_index]

        if bounding_box is not None:
            pts = bounding_box.astype(int)
            pts = pts.reshape((-1, 1, 2))

        else:
            x, y = center_point
            square_coords = [
                (x - 1, y - 1),
                (x + 1, y - 1),
                (x + 1, y + 1),
                (x - 1, y + 1),
                (x - 1, y - 1)
            ]

            # bbox = plt.Polygon(square_coords, closed=True)
            pts = np.array(square_coords, np.int32)
            # pts = bounding_box.astype(int)
            pts = pts.reshape((-1, 1, 2))

        heading = track['heading'][current_index]
        x_velocity = track["xVelocity"][current_index]
        y_velocity = track["yVelocity"][current_index]
        x_acceleration = track["xAcceleration"][current_index]
        y_acceleration = track["yAcceleration"][current_index]

        # Normalize the center point to the range [0, 1]
        center_point = [center_point[0] / width_height[0], center_point[1] / width_height[1]]

        return {
            "pts": pts,
            "center": center_point,
            "object_class": object_class,
            "heading": heading,
            "xVelocity": x_velocity,
            "yVelocity": y_velocity,
            "xAcceleration": x_acceleration,
            "yAcceleration": y_acceleration
        }

    def _get_current_index(self, track_meta, t: int) -> int:
        initial_frame = track_meta["initialFrame"].item()
        current_index = t - initial_frame
        return current_index

    def _get_heading(self, track, track_meta, t) -> float:
        current_index = self._get_current_index(track_meta, t)
        if current_index < 0:
            return None

        return track["heading"][current_index]


    def _extract_historical_data(self, ego_vehicle_track_idx, scene_id, current_frame):
        ego_track = self.tracks[(self.tracks["trackId"] == ego_vehicle_track_idx)
                                & (self.tracks["recordingId"] == scene_id)].iloc[0].to_dict()
        ego_track_meta = self.tracks_meta[(self.tracks_meta["trackId"] == ego_vehicle_track_idx)
                                          & (self.tracks_meta["recordingId"] == scene_id)].iloc[0].to_dict()

        # iterate through all the historical frames upto the current_frame
        historical_adjacent_obs, historical_ego_obs, map_obs = {}, [], []

        backgrond_img = self.background_images[scene_id]
        starting_frame = current_frame - self.history_length
        for t in range(starting_frame, current_frame):  # T, N, D
            map = deepcopy(backgrond_img)
            visible_track_ids = list(self.visibility_data[(self.visibility_data["trackId"] == ego_vehicle_track_idx)
                                                          & (self.visibility_data["frame"] == t)
                                                          & (self.visibility_data["visibility"] == True)
                                                          & (self.visibility_data["recordingId"] == scene_id)
                                                          & (self.visibility_data["located"] == 'FRONT')]
                                     ["adjacentTrackId"].drop_duplicates())

            # Extract historical observations for visible tracks
            recorded_tack_ids = []
            for track_idx in visible_track_ids:
                track = self.tracks[(self.tracks["trackId"] == track_idx)
                                    & (self.tracks["recordingId"] == scene_id)].iloc[0].to_dict()
                track_meta = self.tracks_meta[(self.tracks_meta["trackId"] == track_idx)
                                              & (self.tracks_meta["recordingId"] == scene_id)].iloc[0].to_dict()

                track_info = self._extract_track_info(track, track_meta, t, backgrond_img.shape)
                cv2.fillPoly(map, [track_info["pts"]], (150, 150, 50))  # 150, 100, 150

                track_data = np.array([track_info["center"][0], track_info["center"][1], track_info["heading"],
                                       track_info["xVelocity"], track_info["yVelocity"], track_info["xAcceleration"],
                                       track_info["yAcceleration"]])

                if track_idx not in historical_adjacent_obs.keys():
                    if t > starting_frame:
                        # This object appeared lately. So have to add nulls/empty/zeros for previous frames
                        historical_adjacent_obs[track_idx] = [[0] * self.num_features for _ in range(starting_frame, t)]
                        historical_adjacent_obs[track_idx].append(track_data)
                    else:
                        historical_adjacent_obs[track_idx] = [track_data]
                else:
                    historical_adjacent_obs[track_idx].append(track_data)

                recorded_tack_ids.append(track_idx)

            # Fill the historical observations with zeros for the tracks that are not visible in the current frame
            for track_idx in historical_adjacent_obs.keys():
                if track_idx not in recorded_tack_ids:
                    historical_adjacent_obs[track_idx].append([0])

            # Extract historical observations for the ego vehicle
            track_info = self._extract_track_info(ego_track, ego_track_meta, t, backgrond_img.shape)
            cv2.fillPoly(map, [track_info["pts"]], (255, 180, 200))
            historical_ego_obs.append(np.array([track_info["center"][0],
                                                track_info["center"][1],
                                                track_info["heading"],
                                                track_info["xVelocity"],
                                                track_info["yVelocity"],
                                                track_info["xAcceleration"],
                                                track_info["yAcceleration"]]))

            map_resized = cv2.resize(map, (224, 224), interpolation=cv2.INTER_AREA)
            map_obs.append(map_resized)

        return historical_adjacent_obs, historical_ego_obs, map_obs

    def _extract_ground_truth_data(self, ego_vehicle_track_idx, scene_id, current_frame):
        all_tracks_visibility_df = self.visibility_data[(self.visibility_data["frame"] == current_frame)
                                                        & (self.visibility_data["trackId"] == ego_vehicle_track_idx)
                                                        & (self.visibility_data["recordingId"] == scene_id)
                                                        & (self.visibility_data["located"] == 'FRONT')]

        all_track_idx = all_tracks_visibility_df["adjacentTrackId"].drop_duplicates().tolist()
        visible_track_idx = (all_tracks_visibility_df[all_tracks_visibility_df["visibility"] == True]["adjacentTrackId"]
                             .drop_duplicates().tolist())
        all_tracks = self.tracks[(self.tracks["trackId"].isin(all_track_idx))
                                 & (self.tracks["recordingId"] == scene_id)].to_dict('records')

        visible_track_pts = []
        hidden_tracks_pts = []
        for track in all_tracks:
            track_meta = self.tracks_meta[(self.tracks_meta["trackId"] == track["trackId"])
                                          & (self.tracks_meta["recordingId"] == scene_id)].iloc[0].to_dict()
            pts = self._extract_track_info(track, track_meta, current_frame, self.background_images[scene_id].shape)["pts"].squeeze()

            if track["trackId"] in visible_track_idx:
                visible_track_pts.append(pts)
            else:
                hidden_tracks_pts.append(pts)

            # cv2.fillPoly(backgrond_img, [pts], (0, 255, 255))

        # cv2.imshow('Image with Polygon', backgrond_img)
        return visible_track_pts, hidden_tracks_pts
