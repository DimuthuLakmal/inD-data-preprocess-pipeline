import random
from copy import deepcopy

import numpy
import torch
from torch.utils.data import Dataset
from typing import Tuple

from src.utils.ogm_util import create_OGM_ego
from src.utils.tracks_import import read_from_csv

import json
import sys
import cv2
import numpy as np
from pathlib import Path
import re
import os
import math

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

        start_scene = 0
        end_scene = 27
        filename = "data.pkl"

        self.data_dict = {}

        # Check data file exists
        index_file_path = Path(os.path.join(self.input_path, filename))
        if index_file_path.exists():
            logger.info("Loading index map from {}", index_file_path)
            self.data_dict = pickle.load(open(index_file_path, "rb"))

            # Loading background images
            for scene_id in scene_ids:
                if int(scene_id) < start_scene or int(scene_id) > end_scene:
                    continue

                # Store background images for scenes
                bg_path = os.path.join(self.input_path, 'semantic_maps', f"{scene_id}_background.png")
                img = cv2.imread(bg_path)
                self.background_images[int(scene_id)] = img

        else:
            for scene_id in scene_ids:

                if int(scene_id) < start_scene or int(scene_id) > end_scene:
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
            self.num_features = config[
                'num_features']  # x, y, heading, xVelocity, yVelocity, xAcceleration, yAcceleration, t

            # Load dataset specific visualization parameters from file
            dataset_params_path = Path(config["visualizer_params_dir"]) / "visualizer_params.json"

            if not dataset_params_path.exists():
                logger.error("Could not find dataset visualization parameters in {}", dataset_params_path)
                sys.exit(-1)

            with open(dataset_params_path) as f:
                self.dataset_params = json.load(f)

            if self.dataset not in self.dataset_params["datasets"]:
                logger.error(
                    "Visualization parameters for dataset {} not found in {}. Please make sure, that the needed "
                    "parameters are given", self.dataset, dataset_params_path)
                sys.exit(-1)

            self.dataset_params = self.dataset_params["datasets"][self.dataset]
            self.scale_down_factor = self.dataset_params["scale_down_factor"]

            self.tracks = pd.concat(self.tracks, ignore_index=True)
            self.tracks_meta = pd.concat(self.tracks_meta, ignore_index=True)
            self.visibility_data = pd.concat(self.visibility_data, ignore_index=True)

            for scene_id in scene_ids:
                scene_id = int(scene_id)
                if scene_id < start_scene or scene_id > end_scene:
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
                for i_frame in range(minimum_frame + self.history_length, maximum_frame,
                                     (self.history_length)):

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

                    # Extract ego vehicle track information
                    ego_vehicle_track_idx = random.choice(eligible_tracks)
                    ego_track = self.tracks[(self.tracks["trackId"] == ego_vehicle_track_idx)
                                            & (self.tracks["recordingId"] == scene_id)].iloc[0].to_dict()
                    ego_track_meta = self.tracks_meta[(self.tracks_meta["trackId"] == ego_vehicle_track_idx)
                                                      & (self.tracks_meta["recordingId"] == scene_id)].iloc[0].to_dict()

                    pts_ego = self._extract_track_info(ego_track, ego_track_meta, i_frame,
                                                       self.background_images[scene_id].shape)["pts"]
                    heading_ego = self._get_heading(ego_track, ego_track_meta, i_frame)

                    # extract historical data for the ego vehicle
                    historical_adjacent_obs, historical_ego_obs, map_obs, visible_tracks_pts, last_recorded_t = (
                        self._extract_historical_data(ego_track, ego_track_meta, scene_id, i_frame))

                    # extract ground truth data for the ego vehicle
                    hidden_tracks_pts = self._extract_ground_truth_data(ego_track, ego_track_meta, scene_id, i_frame)

                    # Create OGM
                    ogm, ogm_gt, hidden_ogm_cells, hidden_cell_polygon_xys = create_OGM_ego(pts_ego.squeeze(),
                                                                                            heading_ego,
                                                                                            visible_tracks_pts,
                                                                                            hidden_tracks_pts,
                                                                                            self.background_images[
                                                                                                scene_id],
                                                                                            self.fixed_blocks_info[
                                                                                                scene_id])

                    if ogm is None:
                        continue

                    # Check if there are any hidden ogm cells. Should have at least one hidden ogm cell to create a valid sample
                    if len(hidden_ogm_cells) == 0:
                        continue

                    # Extract distances for hidden ogm cells from adjacent tracks (This is a bi-partition graph)
                    edge_weights, edge_index = self._extract_edge_info(historical_adjacent_obs, hidden_ogm_cells,
                                                                       last_recorded_t)

                    self.data_dict[str(scene_id) + '_' + str(i_frame) + '_' + str(ego_vehicle_track_idx)] = {
                        "historical_adjacent_obs": historical_adjacent_obs,
                        "historical_ego_obs": historical_ego_obs,
                        "map_obs": None,
                        "hidden_tracks_pts": hidden_tracks_pts,
                        "visible_tracks_pts": visible_tracks_pts,
                        "edge_weights": edge_weights,
                        "edge_index": edge_index,
                        "ogm": ogm,
                        "ogm_gt": ogm_gt,
                        "hidden_ogm_cells": hidden_ogm_cells,
                        "hidden_cell_polygon_xys": hidden_cell_polygon_xys
                    }

                    print(str(scene_id) + '_' + str(i_frame) + '_' + str(ego_vehicle_track_idx))

            pickle.dump(self.data_dict, open(index_file_path, "wb"))

        self.keys = list(self.data_dict.keys())
        print("Done Loading")

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        key = self.keys[idx]
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
        historical_adjacent_obs, historical_ego_obs, map_obs, hidden_tracks_pts, visible_tracks_pts, edge_weights, \
            edge_index, ogm, ogm_gt, hidden_cell_polygon_xys = (data_dict["historical_adjacent_obs"],
                                       data_dict["historical_ego_obs"],
                                       data_dict["map_obs"],
                                       data_dict["hidden_tracks_pts"],
                                       data_dict["visible_tracks_pts"],
                                       data_dict["edge_weights"],
                                       data_dict["edge_index"],
                                       data_dict["ogm"],
                                       data_dict["ogm_gt"],
                                       data_dict["hidden_cell_polygon_xys"])

        # Create a black background image a size of background_img
        blank_img = np.zeros_like(backgrond_img[:, :, 0:1])  # Create a single channel image
        for cel in hidden_cell_polygon_xys:
            cv2.fillPoly(blank_img, [np.array(cel).astype(np.int32)], (255, 255, 255))

        # Extract Ground Truth
        # ego_track = self.tracks[(self.tracks["trackId"] == ego_vehicle_track_idx)
        #                         & (self.tracks["recordingId"] == scene_id)].iloc[0].to_dict()
        # ego_track_meta = self.tracks_meta[(self.tracks_meta["trackId"] == ego_vehicle_track_idx)
        #                                   & (self.tracks_meta["recordingId"] == scene_id)].iloc[0].to_dict()
        # pts_ego = self._extract_track_info(ego_track, ego_track_meta, current_frame, backgrond_img.shape)["pts"]
        # heading_ego = self._get_heading(ego_track, ego_track_meta, current_frame)
        #
        # # Create OGM
        # ogm, ogm_gt = create_OGM_ego(pts_ego.squeeze(), heading_ego, visible_tracks_pts, hidden_tracks_pts,
        #                              gt_background_img, self.fixed_blocks_info[scene_id])
        #
        # # Remove distance feature from historical_adjacent_obs
        # historical_adjacent_obs = np.array(list(historical_adjacent_obs.values()), dtype=np.float32)
        # historical_adjacent_no_e = historical_adjacent_obs[:, :, :-1]
        #
        # # Extract the edge weights from the historical_adjacent_obs
        # edge_weights = np.expand_dims(historical_adjacent_obs[:, :, -1], axis=-1)


        historical_adjacent_obs = np.array(list(historical_adjacent_obs.values()), dtype=np.float32)
        historical_adjacent_no_e = historical_adjacent_obs[:, :, :-1]

        hidden_ogm_cells = np.array(data_dict["hidden_ogm_cells"], dtype=np.float32)

        # Visual representation of the map
        map_resized = cv2.resize(self.background_images[scene_id], (224, 224), interpolation=cv2.INTER_AREA)
        hidden_cells_resized = cv2.resize(blank_img, (224, 224), interpolation=cv2.INTER_AREA)

        # Normalize map and hiddden_cells_resized
        map_resized = map_resized / 255.0  # Normalize to [0, 1]
        hidden_cells_resized = hidden_cells_resized / 255.0  # Normalize to [0, 1]

        historical_adjacent_no_e[:, :, 2:3] = historical_adjacent_no_e[:, :, 2:3] / 360.0  # Normalize heading to [0, 1]. This is a mistake done when extracting the data
        historical_adjacent_no_e[:, :, 3:] = historical_adjacent_no_e[:, :, 3:] / 10.0
        seq_mask = np.all(historical_adjacent_no_e == 0, axis=-1)  # Create a sequence mask where all features are zeros

        input = {
            "historical_adjacent_obs": historical_adjacent_no_e,
            "historical_ego_obs": np.array(historical_ego_obs, dtype=np.float32),
            "map_obs": map_resized.astype(np.float32),
            "ogm": ogm.astype(np.float32),
            "edge_weights": np.expand_dims(numpy.array(edge_weights, dtype=np.float32), axis=-1),
            "edge_index": numpy.array(edge_index, dtype=np.int64),
            "hidden_ogm_cells": hidden_ogm_cells[:, :-1],
            "hidden_cells_resized": hidden_cells_resized.astype(np.float32),
            "seq_mask": seq_mask,
            "keys": keys,
            "background_image": gt_background_img
        }
        target = hidden_ogm_cells[:, -1:].astype(np.float32)

        return input, target

    def _extract_track_info(self, track, track_meta, t: int, height_width: tuple) -> dict:
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
        center_point = [center_point[0] / height_width[1], center_point[1] / height_width[0]]

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

    def _extract_historical_data(self, ego_track, ego_track_meta, scene_id, current_frame):
        ego_vehicle_track_idx = ego_track["trackId"]
        # iterate through all the historical frames upto the current_frame
        historical_adjacent_obs, historical_ego_obs, map_obs, visible_tracks_pts, last_recorded_t = {}, [], [], [], {}

        backgrond_img = self.background_images[scene_id]
        starting_frame = current_frame - self.history_length

        history_t = 0
        for t in range(starting_frame, current_frame + 1):  # T, N, D
            map = deepcopy(backgrond_img)

            # Extract historical observations for the ego vehicle
            ego_track_info = self._extract_track_info(ego_track, ego_track_meta, t, backgrond_img.shape)
            cv2.fillPoly(map, [ego_track_info["pts"]], (255, 180, 200))
            historical_ego_obs.append(np.array([ego_track_info["center"][0],
                                                ego_track_info["center"][1],
                                                ego_track_info["heading"],
                                                ego_track_info["xVelocity"],
                                                ego_track_info["yVelocity"],
                                                ego_track_info["xAcceleration"],
                                                ego_track_info["yAcceleration"]]))

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

                # Calculate distance from the ego vehicle to the track center
                distance_ego = math.sqrt((track_info["center"][0] - ego_track_info["center"][0]) ** 2 +
                                         (track_info["center"][1] - ego_track_info["center"][1]) ** 2)

                track_data = np.array([track_info["center"][0], track_info["center"][1], track_info["heading"],
                                       track_info["xVelocity"], track_info["yVelocity"], track_info["xAcceleration"],
                                       track_info["yAcceleration"], (history_t / self.history_length), distance_ego])

                last_recorded_t[track_idx] = history_t  # Store the last recorded time for the track

                # visible track points
                if t == current_frame:
                    visible_tracks_pts.append(np.squeeze(track_info["pts"]))

                if track_idx not in historical_adjacent_obs.keys():
                    if t > starting_frame:
                        # This object appeared lately. So have to add nulls/empty/zeros for previous frames
                        historical_adjacent_obs[track_idx] = [np.zeros(self.num_features + 1) for _ in
                                                              range(starting_frame, t)]  # +1 for distance
                        historical_adjacent_obs[track_idx].append(track_data)
                    else:
                        historical_adjacent_obs[track_idx] = [track_data]
                else:
                    historical_adjacent_obs[track_idx].append(track_data)

                recorded_tack_ids.append(track_idx)

            # Fill the historical observations with zeros for the tracks that are not visible in the current frame
            for track_idx in historical_adjacent_obs.keys():
                if track_idx not in recorded_tack_ids:
                    historical_adjacent_obs[track_idx].append(np.zeros(self.num_features + 1))  # +1 for distance

            # map_resized = cv2.resize(map, (224, 224), interpolation=cv2.INTER_AREA)
            # map_obs.append(map_resized)

            history_t += 1

        if len(historical_adjacent_obs) != len(visible_tracks_pts):
            logger.warning("Number of historical adjacent observations does not match with visible tracks points. "
                           "This might be due to missing tracks in the visibility data.")

        return historical_adjacent_obs, historical_ego_obs, map_obs, visible_tracks_pts, last_recorded_t

    def _extract_ground_truth_data(self, ego_track, ego_track_meta, scene_id, current_frame):
        ego_vehicle_track_idx = ego_track["trackId"]

        hidden_tracks_visibility_df = self.visibility_data[(self.visibility_data["frame"] == current_frame)
                                                           & (self.visibility_data["trackId"] == ego_vehicle_track_idx)
                                                           & (self.visibility_data["recordingId"] == scene_id)
                                                           & (self.visibility_data["visibility"] == False)
                                                           & (self.visibility_data["located"] == 'FRONT')]

        hidden_track_idx = hidden_tracks_visibility_df["adjacentTrackId"].drop_duplicates().tolist()
        hidden_tracks = self.tracks[(self.tracks["trackId"].isin(hidden_track_idx))
                                    & (self.tracks["recordingId"] == scene_id)].to_dict('records')

        ego_track_info = self._extract_track_info(ego_track, ego_track_meta, current_frame,
                                                  self.background_images[scene_id].shape)

        hidden_tracks_pts = []
        for track in hidden_tracks:
            track_meta = self.tracks_meta[(self.tracks_meta["trackId"] == track["trackId"])
                                          & (self.tracks_meta["recordingId"] == scene_id)].iloc[0].to_dict()
            track_info = self._extract_track_info(track, track_meta, current_frame,
                                                  self.background_images[scene_id].shape)

            # Calculate distance from the ego vehicle to the track center
            distance = math.sqrt((track_info["center"][0] - ego_track_info["center"][0]) ** 2 +
                                 (track_info["center"][1] - ego_track_info["center"][1]) ** 2)
            pts = np.squeeze(track_info["pts"])

            # Create a numpy array with the track information
            track_data = np.array([track_info["center"][0], track_info["center"][1], track_info["heading"],
                                   track_info["xVelocity"], track_info["yVelocity"], track_info["xAcceleration"],
                                   track_info["yAcceleration"], distance])

            hidden_tracks_pts.append(np.squeeze(pts))

            # cv2.fillPoly(backgrond_img, [pts], (0, 255, 255))

        # cv2.imshow('Image with Polygon', backgrond_img)
        return hidden_tracks_pts

    def _extract_edge_info(self, historical_adjacent_obs, hidden_ogm_cells, last_recorded_t) -> Tuple[list, list]:
        edge_weights = []
        edge_src, edge_dst = [], []
        for i, cell in enumerate(hidden_ogm_cells):
            for j, (track_idx, obs) in enumerate(historical_adjacent_obs.items()):
                # Calculate the distance from the cell to the track center
                obs_last_t = last_recorded_t[track_idx]
                distance = math.sqrt((cell[0] - obs[obs_last_t][0]) ** 2 + (cell[1] - obs[obs_last_t][1]) ** 2)
                edge_weights.append(distance)

                edge_src.append(j)  # Source index is the track index
                edge_dst.append(i)  # Destination index is the cell index

        edge_index = [edge_src, edge_dst]
        return edge_weights, edge_index
