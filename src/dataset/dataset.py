import random
from copy import deepcopy

import numpy
import torch
from torch.utils.data import Dataset
from typing import Tuple

from utils.ogm_util import create_OGM_ego, get_vert

import json
import cv2
import numpy as np
from pathlib import Path
import re
import os
import math

from loguru import logger
import pickle


class OGMDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, config, phase):

        scene_ids = set()
        for fname in os.listdir(config['dataset_dir']):
            match = re.match(r"(\d+)_.*\.csv", fname)
            if match:
                scene_ids.add(match.group(1))

        scene_ids = sorted(scene_ids)
        print(f"Found scenes: {scene_ids}")

        self.input_path = config['dataset_dir']
        self.annotations_path = config['label_dir'] + '/' + phase

        self.history_length = config["history_length"]

        self.tracks = []
        self.tracks_meta = []
        self.visibility_data = []
        self.background_images = {}
        self.fixed_blocks_info = {}
        self.frame_to_track_idxs = {}
        self.class_dict = {'car': 0, 'truck_bus': 1, 'bicycle': 2, 'pedestrian': 3}

        # You can adjust if you don't want to use all scenes for training or testing.
        start_scene = config['start_scene']
        end_scene = config['end_scene']
        filename = config['observation_data_filename']  # if your observation scattered in multiple files, please merge them.

        self.data_dict = {}

        # Check data file exists
        observation_file_path = Path(os.path.join(self.input_path, filename))
        logger.info("Loading Observations and OGM data from {}", observation_file_path)
        self.data_dict = pickle.load(open(observation_file_path, "rb"))

        # Loading background images
        for scene_id in scene_ids:
            if int(scene_id) < start_scene or int(scene_id) > end_scene:
                continue

            # Store background images for scenes
            bg_path = os.path.join(self.input_path, 'semantic_maps', f"{scene_id}_background.png")
            img = cv2.imread(bg_path)
            self.background_images[int(scene_id)] = img

        # load json files from annotations path
        annotation_files = [f for f in os.listdir(self.annotations_path) if f.endswith('.json')]

        label_dict = {}

        # Creating a balanced dataset
        # First find out training data with positive cells
        files_with_positive_cells = []
        for file in annotation_files:
            key = file.split('.')[0]
            with open(os.path.join(self.annotations_path, file), 'r') as f:
                data = json.load(f)
                for cell in data:
                    if cell['label'] == 1:
                        files_with_positive_cells.append(key)
                        break

        random_positive_keys = np.random.choice(files_with_positive_cells,
                                                min(int(len(annotation_files)/2), len(files_with_positive_cells)),
                                                replace=False)

        for file in annotation_files:
            key = file.split('.')[0]
            scene_id = int(key.split('_')[0])

            with open(os.path.join(self.annotations_path, file), 'r') as f:
                data = json.load(f)
                normalised_data = []
                hidden_ogm_cells_xys = []
                for cell in data:
                    normalised_data.append([cell['cx'] / self.background_images[scene_id].shape[1],
                                           cell['cy'] / self.background_images[scene_id].shape[0],
                                           cell['label']])

                    # observation data for ego vehicle at current frame also stored using the same key in data_dict
                    ego_vehicle_data = self.data_dict[key]["historical_ego_obs"][-1]
                    hidden_ogm_cell_xy = get_vert(cell['cx'], cell['cy'], ego_vehicle_data[2], length=20.0, width=20.0)
                    hidden_ogm_cells_xys.append(hidden_ogm_cell_xy)

                if len(normalised_data) == 0:  # No hidden cells selected. Not sure if this is needed anymore
                    continue

                label_dict[key] = (normalised_data, hidden_ogm_cells_xys)

        self.label_dict = label_dict

        keys = list(label_dict.keys())  # These are the frame keys selected for training/testing
        for key in keys:
            data_dict = self.data_dict[key]
            ogm_cells, ogm_cells_xys = label_dict[key]

            # check how many adjacent agents are there
            historical_adjacent_obs = data_dict["historical_adjacent_obs"]
            num_adjacent_agents = len(historical_adjacent_obs.keys())

            # if there are less than 5 adjacent agents, remove that entry from label dict and data dict
            # Find moving agents. This is only used in ablation studies
            # historical_obs = np.array((list(historical_adjacent_obs.values())))
            # speeds_x = historical_obs[:, :, 3]  # Assuming speed in x is at index 3
            # speeds_y = historical_obs[:, :, 4]  # Assuming speed in y is at index 4
            # # if any agent has non-zero speed at any time step, consider it moving
            # moving_agents = np.where(np.any((speeds_x != 0) | (speeds_y != 0), axis=1))[0]
            # num_moving_agents = len(moving_agents)

            if key in random_positive_keys:
                positive_indices = []
                for i, cell in enumerate(ogm_cells):
                    if cell[2] == 1:  # checking the label
                        positive_indices.append(i)

                random_index = random.choice(positive_indices)

            else:
                random_index = random.randint(0, len(label_dict[key][0]) - 1)

            ogm_cells = [ogm_cells[random_index]]
            ogm_cells_xys = [ogm_cells_xys[random_index]]

            historical_adjacent_obs, hidden_ogm_cells = (data_dict["historical_adjacent_obs"], data_dict["hidden_ogm_cells"])
            last_recorded_t = {}
            for i, (veh_index, obs) in enumerate(historical_adjacent_obs.items()):
                # Find the index of the last non-zero observation obs np array
                mask = np.any(np.array(obs) != 0, axis=1)
                last_t = np.where(mask)[0].max() if np.any(mask) else None
                last_recorded_t[veh_index] = last_t

            # Extract distances for hidden ogm cells from adjacent tracks (This is a bi-partition graph)
            edge_weights, edge_index = self._extract_edge_info(historical_adjacent_obs, ogm_cells,
                                                               last_recorded_t)

            data_dict["edge_weights"] = edge_weights
            data_dict["edge_index"] = edge_index
            data_dict["hidden_ogm_cells"] = np.array(ogm_cells, dtype=np.float32)
            data_dict["hidden_cell_polygon_xys"] = np.array(ogm_cells_xys, dtype=np.float32)
            self.data_dict[key] = data_dict

        self.keys = list(self.label_dict.keys())
        print("Done Loading")

    def __len__(self):
        return len(self.label_dict)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        key = self.keys[idx]
        keys = key.split("_")
        scene_id = int(keys[0])

        data_dict = self.data_dict[key]

        # map
        backgrond_img = self.background_images[scene_id]
        gt_background_img = deepcopy(backgrond_img)

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
            cv2.fillPoly(gt_background_img, [np.array(cel).astype(np.int32)], (0, 102, 204))

        historical_adjacent_obs = np.array(list(historical_adjacent_obs.values()), dtype=np.float32)

        hidden_ogm_cells = np.array(data_dict["hidden_ogm_cells"], dtype=np.float32)

        # Visual representation of the map
        map_resized = cv2.resize(gt_background_img, (224, 224), interpolation=cv2.INTER_AREA)
        hidden_cells_resized = cv2.resize(blank_img, (224, 224), interpolation=cv2.INTER_AREA)

        seq_mask = np.all(historical_adjacent_obs == 0, axis=-1)  # Create a sequence mask where all features are zeros

        # Fixing a class type issue (0 is used to represent car type. Replacing 0 with 4)
        veh_type = historical_adjacent_obs[..., 7]  # (B, N, T)
        mask = (veh_type == 0) & (~seq_mask)
        veh_type[mask] = 4
        historical_adjacent_obs[..., 7] = veh_type

        # Attaching scene_id as a feature
        scene_id_norm = scene_id / 10  # will be divided it further later to bring the range of 0 and 1
        scene_id_arr = np.full(historical_adjacent_obs.shape[:-1] + (1,), scene_id_norm,
                               dtype=historical_adjacent_obs.dtype)  # (B, N, T, 1)
        historical_adjacent_obs = np.concatenate([historical_adjacent_obs, scene_id_arr], axis=-1)

        historical_adjacent_obs[:, :, 2:3] = historical_adjacent_obs[:, :,
                                             2:3] / 360.0  # Normalize heading to [0, 1]. This is a mistake done when extracting the data
        historical_adjacent_obs[:, :, 3:] = historical_adjacent_obs[:, :, 3:] / 10.0

        historical_adjacent_input = np.concatenate((historical_adjacent_obs[:, :, :9],
                                                    historical_adjacent_obs[:, :, 10:11]), axis=-1)

        input = {
            "historical_adjacent_obs": historical_adjacent_input,
            "historical_ego_obs": np.array(historical_ego_obs, dtype=np.float32),
            "map_obs": map_resized.astype(np.float32),
            "ogm": ogm.astype(np.float32),
            "edge_weights": np.expand_dims(numpy.array(edge_weights, dtype=np.float32), axis=-1),
            "edge_index": numpy.array(edge_index, dtype=np.int64),
            "hidden_ogm_cells": hidden_ogm_cells[:, :-1],
            "hidden_cells_resized": hidden_cells_resized.astype(np.float32),
            "seq_mask": seq_mask,
            "scene_id": np.array([scene_id], np.float32)
        }
        target = hidden_ogm_cells[:, -1:].astype(np.float32)

        return input, target


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
