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


class TrackDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, config, data_path):

        scene_ids = set()
        for fname in os.listdir(data_path):
            match = re.match(r"(\d+)_.*\.csv", fname)
            if match:
                scene_ids.add(match.group(1))

        scene_ids = sorted(scene_ids)
        print(f"Found scenes: {scene_ids}")

        self.input_path = data_path

        self.tracks = []
        self.tracks_meta = []
        self.visibility_data = []
        self.background_images = {}
        self.fixed_blocks_info = {}
        self.frame_to_track_idxs = {}
        self.index_map = {}

        for scene_id in scene_ids:

            if int(scene_id) > 1:
                break

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

            # Prefix trackIds & adjacentTrackIds
            # scene_prefix = f"{scene_id}_"
            # tracks["trackId"] = tracks["trackId"].astype(str).apply(lambda x: scene_prefix + x)
            # tracks_meta["trackId"] = tracks_meta["trackId"].astype(str).apply(lambda x: scene_prefix + x)
            # visibility_df["trackId"] = visibility_df["trackId"].astype(str).apply(lambda x: scene_prefix + x)
            # visibility_df["adjacentTrackId"] = visibility_df["adjacentTrackId"].astype(str).apply(lambda x: scene_prefix + x)

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

        self.index_map = {}
        # Check index file saved into a file
        index_file_path = Path(os.path.join(self.input_path, "index_map.pkl"))
        if index_file_path.exists():
            logger.info("Loading index map from {}", index_file_path)
            self.index_map = pickle.load(open(index_file_path, "rb"))
        else:
            for scene_id in scene_ids:
                scene_id = int(scene_id)
                if scene_id > 1:
                    break

                tracks_meta = self.tracks_meta[(self.tracks_meta["recordingId"] == scene_id)]
                visibility_data = self.visibility_data[(self.visibility_data["recordingId"] == scene_id)]

                # Determine the first and last frame
                minimum_frame = tracks_meta["initialFrame"].min()
                maximum_frame = tracks_meta["finalFrame"].max()

                # Create a mapping between frame and idxs of tracks for quick lookup during playback
                frame_to_track_idxs = {}
                for i_frame in range(minimum_frame, maximum_frame + 1):
                    indices = tracks_meta[(tracks_meta["initialFrame"] <= i_frame) & (tracks_meta["finalFrame"] >= i_frame)]["trackId"].tolist()
                    frame_to_track_idxs[i_frame] = indices

                # We have to find out timesteps that have atleast one hidden record in the visiblity data.
                # We cannot start from the minimum_frame as we have to include the history as well.
                for i_frame in range(minimum_frame + self.history_length, maximum_frame + 1):
                    if i_frame % 4 == 0:
                        hidden_objects = visibility_data[(visibility_data['frame'] == i_frame) & (visibility_data['visibility'] == False)]

                        if len(hidden_objects) == 0: continue

                        tracks_with_full_history = []
                        for track_idx in frame_to_track_idxs[i_frame]:
                            initial_frame = tracks_meta[tracks_meta["trackId"] == track_idx]["initialFrame"].item()
                            vehicle_type = tracks_meta[tracks_meta["trackId"] == track_idx]["class"].item()
                            if initial_frame <= (i_frame - self.history_length) and (vehicle_type == 'car'):
                                tracks_with_full_history.append(track_idx)

                        tracks_with_limited_visibility = list(hidden_objects["trackId"].drop_duplicates())
                        # Find intersection of tracks with limited visibility and tracks with full history
                        eligible_tracks = list(set(tracks_with_limited_visibility) & set(tracks_with_full_history))
                        if len(eligible_tracks) == 0: continue

                        ego_vehicle_track_idx = random.choice(eligible_tracks)
                        self.index_map[str(scene_id) + '_' + str(i_frame)] = ego_vehicle_track_idx

                pickle.dump(self.index_map, open(index_file_path, "wb"))

                        # Deriving Occupancy Grid Map for the ego vehicle
                        # Create grid map cell polygons in front of the ego vehicle


        print("Done Loading")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        bb_boxes = []
        key = list(self.index_map.keys())[idx]
        scene_id = int(key.split("_")[0])
        current_frame = int(key.split("_")[1])

        ego_vehicle_track_idx = self.index_map[key]
        ego_track = self.tracks[(self.tracks["trackId"] == ego_vehicle_track_idx)
                                & (self.tracks["recordingId"] == scene_id)].iloc[0].to_dict()
        ego_track_meta = self.tracks_meta[(self.tracks_meta["trackId"] == ego_vehicle_track_idx)
                                          & (self.tracks_meta["recordingId"] == scene_id)].iloc[0].to_dict()

        # iterate through all the historical frames upto the current_frame
        historical_adjacent_obs, historical_ego_obs, map_obs = {}, [], []

        backgrond_img = self.background_images[scene_id]
        starting_frame = current_frame - self.history_length
        for t in range(starting_frame, current_frame): # T, N, D
            map = deepcopy(backgrond_img)
            visible_track_ids = list(self.visibility_data[(self.visibility_data["trackId"] == ego_vehicle_track_idx)
                                                          & (self.visibility_data["frame"] == t)
                                                          & (self.visibility_data["visibility"] == True)
                                                          & (self.visibility_data["recordingId"] == scene_id)]
                                     ["adjacentTrackId"].drop_duplicates())

            # Extract historical observations for visible tracks
            recorded_tack_ids = []
            for track_idx in visible_track_ids:
                track = self.tracks[(self.tracks["trackId"] == track_idx)
                                    & (self.tracks["recordingId"] == scene_id)].iloc[0].to_dict()
                track_meta = self.tracks_meta[(self.tracks_meta["trackId"] == track_idx)
                                              & (self.tracks_meta["recordingId"] == scene_id)].iloc[0].to_dict()

                track_info = self._extract_track_info(track, track_meta, t, backgrond_img.shape)
                cv2.fillPoly(map, [track_info["pts"]], (150, 150, 50)) #150, 100, 150

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

            # cv2.imshow('Image with Polygon', map)

        # cv2.imwrite("history.png", map)
        # Extract Ground Truth
        # Find hidden tracks for the selected ego vehicle
        hidden_track_idx = list(self.visibility_data[(self.visibility_data["frame"] == current_frame)
                                                     & (self.visibility_data["visibility"] == False)
                                                     & (self.visibility_data["trackId"] == ego_vehicle_track_idx)
                                                     & (self.visibility_data["recordingId"] == scene_id)]
                                ["adjacentTrackId"].drop_duplicates())
        hidden_tracks = self.tracks[(self.tracks["trackId"].isin(hidden_track_idx))
                                    & (self.tracks["recordingId"] == scene_id)]
        # TODO: Do we need to filter out the tracks that are not visible in the current frame?
        #  It seems we don't need to do that as it's already done in the visibility data.
        hidden_tracks["existsInFrame"] = hidden_tracks["frame"].apply(lambda x: current_frame in x)
        hidden_tracks = hidden_tracks[hidden_tracks["existsInFrame"]].to_dict('records')

        pts_ego = self._extract_track_info(ego_track, ego_track_meta, current_frame, backgrond_img.shape)["pts"]
        heading_ego = self._get_heading(ego_track, ego_track_meta, current_frame)
        cv2.fillPoly(backgrond_img, [pts_ego], (0, 255, 0))
        for h in hidden_tracks:
            hidden_track_meta = self.tracks_meta[(self.tracks_meta["trackId"] == h["trackId"])
                                                 & (self.tracks_meta["recordingId"] == scene_id)].iloc[0].to_dict()

            pts_hidden = self._extract_track_info(h, hidden_track_meta, current_frame, backgrond_img.shape)["pts"]
            cv2.fillPoly(backgrond_img, [pts_hidden], (255, 255, 0))

        # Create OGM
        visible_track_idx = list(self.visibility_data[(self.visibility_data["frame"] == current_frame)
                                                     & (self.visibility_data["visibility"] == True)
                                                     & (self.visibility_data["trackId"] == ego_vehicle_track_idx)
                                                     & (self.visibility_data["recordingId"] == scene_id)]
                                ["adjacentTrackId"].drop_duplicates())
        visible_tracks = self.tracks[(self.tracks["trackId"].isin(visible_track_idx))
                                    & (self.tracks["recordingId"] == scene_id)].to_dict('records')

        visible_pts = []
        for track in visible_tracks:
            track_meta = self.tracks_meta[(self.tracks_meta["trackId"] == track["trackId"])
                                          & (self.tracks_meta["recordingId"] == scene_id)].iloc[0].to_dict()
            pts = self._extract_track_info(track, track_meta, current_frame, backgrond_img.shape)["pts"].squeeze()
            visible_pts.append(pts)

            cv2.fillPoly(backgrond_img, [pts], (0, 255, 255))

        # cv2.imshow('Image with Polygon', backgrond_img)

        ogm, ogm_gt = create_OGM_ego(pts_ego.squeeze(), heading_ego, visible_pts, backgrond_img, self.fixed_blocks_info[scene_id])

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
        initial_frame = track_meta["initialFrame"]
        current_index = t - initial_frame
        return current_index

    def _get_heading(self, track, track_meta, t) -> float:
        current_index = self._get_current_index(track_meta, t)
        if current_index < 0:
            return None

        return track["heading"][current_index]
