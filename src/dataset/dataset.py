import random

import numpy
import torch
from torch.utils.data import Dataset

from src.utils.ogm_util import create_OGM_ego, get_vert
from src.dataset import feature_builder
from src.utils.semantic_maps import semantic_map_to_one_hot

import json
import cv2
import numpy as np
from pathlib import Path
import re
import os

from loguru import logger
import pickle


SEMANTIC_PALETTE_BGR = {
    'fixed_blocks': (0, 255, 0),
    'walking_path': (255, 0, 0),
    'vegetation': (255, 0, 150),
    'road': (0, 0, 255),
    'lines': (255, 255, 255),
    'pedestrian_crossing': (255, 150, 0),
    'parking': (0, 255, 255),
    'refuge_island': (90, 90, 90),
    'background': (0, 0, 0)
}


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
        self.background_images = {}  # retraining only for functioning of old code
        self.semantic_maps = {}
        self.semantic_map_shapes = {}
        self.load_maps(scene_ids, start_scene, end_scene)

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

            if scene_id not in [0, 7, 8, 18, 19]:
                continue

            with open(os.path.join(self.annotations_path, file), 'r') as f:
                data = json.load(f)
                normalised_data = []
                hidden_ogm_cells_xys = []
                for cell in data:
                    normalised_data.append([cell['cx'] / (self.background_images[scene_id].shape[1] -1),
                                           cell['cy'] / (self.background_images[scene_id].shape[0] - 1),
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
            last_recorded_t = feature_builder.compute_last_recorded_t(historical_adjacent_obs)

            # Extract distances for hidden ogm cells from adjacent tracks (This is a bi-partition graph)
            edge_weights, edge_index = feature_builder.extract_edge_info(historical_adjacent_obs, ogm_cells,
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

        data_dict = self.data_dict[key]
        return self._build_sample(key, data_dict["edge_weights"], data_dict["edge_index"],
                                  data_dict["hidden_ogm_cells"], data_dict["hidden_cell_polygon_xys"])

    def get_all_candidate_cells_sample(self, key):
        """
        Builds a model-ready sample using ALL candidate hidden cells recorded for this frame
        (self.label_dict[key], before the single-cell narrowing applied for training), instead
        of the one cell randomly selected in __init__. Intended for inference, e.g. predicting
        occupancy for every candidate cell of a frame in a single forward pass.
        """
        ogm_cells, ogm_cells_xys = self.label_dict[key]
        historical_adjacent_obs = self.data_dict[key]["historical_adjacent_obs"]
        last_recorded_t = feature_builder.compute_last_recorded_t(historical_adjacent_obs)
        edge_weights, edge_index = feature_builder.extract_edge_info(historical_adjacent_obs, ogm_cells, last_recorded_t)

        hidden_ogm_cells = np.array(ogm_cells, dtype=np.float32)
        hidden_cell_polygon_xys = np.array(ogm_cells_xys, dtype=np.float32)

        return self._build_sample(key, edge_weights, edge_index, hidden_ogm_cells, hidden_cell_polygon_xys)

    def _build_sample(self, key, edge_weights, edge_index, hidden_ogm_cells, hidden_cell_polygon_xys):
        keys = key.split("_")
        scene_id = int(keys[0])

        data_dict = self.data_dict[key]

        # map
        backgrond_img = self.background_images[scene_id]

        # Historical observations
        historical_adjacent_obs, historical_ego_obs, map_obs, hidden_tracks_pts, visible_tracks_pts, \
            ogm, ogm_gt = (data_dict["historical_adjacent_obs"],
                          data_dict["historical_ego_obs"],
                          data_dict["map_obs"],
                          data_dict["hidden_tracks_pts"],
                          data_dict["visible_tracks_pts"],
                          data_dict["ogm"],
                          data_dict["ogm_gt"])

        hidden_ogm_cells = np.array(hidden_ogm_cells, dtype=np.float32)

        map_resized = self.semantic_maps[scene_id].copy()
        hidden_cells_resized = cv2.resize(np.zeros_like(backgrond_img[:, :, 0:1]), (224, 224),
                                          interpolation=cv2.INTER_AREA)

        historical_adjacent_input, seq_mask = feature_builder.build_vehicle_tensor(
            historical_adjacent_obs, scene_id)

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

    def load_maps(self, scene_ids, start_scene, end_scene):
        for scene_id in scene_ids:

            scene_id = int(scene_id)

            if (scene_id < start_scene or scene_id > end_scene):
                continue

            bg_path = os.path.join(
                self.input_path,
                "semantic_maps",
                f"{scene_id:02d}_background.png",
            )

            image_bgr = cv2.imread(
                bg_path,
                cv2.IMREAD_COLOR,
            )

            if image_bgr is None:
                raise FileNotFoundError(
                    bg_path
                )

            # Retain this if other existing dataset
            # functions still need the original image.
            self.background_images[scene_id] = image_bgr
            h, w = image_bgr.shape[:2]

            self.semantic_map_shapes[scene_id] = (h, w)

            one_hot, class_ids, class_names = (
                semantic_map_to_one_hot(
                    image_bgr=image_bgr,
                    palette_bgr=SEMANTIC_PALETTE_BGR,
                    output_size=(224, 224),
                )
            )

            # [K,224,224]
            self.semantic_maps[scene_id] = one_hot
