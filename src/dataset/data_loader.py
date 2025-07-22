import random
from copy import deepcopy

import matplotlib
import torch
from torch.utils.data import Dataset

from src.track_visualizer import DataError
from src.utils.tracks_import import read_from_csv

matplotlib.use('qt5agg')

import json
import sys
import os
import cv2
import numpy as np
from pathlib import Path

from loguru import logger
import pandas as pd


class TrackDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, config, visibility_file, recording_meta_file, tracks_meta_file, tracks_file,
                 fixed_blocks_meta_file):
        tracks, tracks_meta, recording_meta, fixed_blocks_info = read_from_csv(tracks_file,
                                                                               tracks_meta_file,
                                                                               recording_meta_file,
                                                                               fixed_blocks_meta_file,
                                                                               include_px_coordinates=True)

        self.visibility_data = pd.read_csv(visibility_file)
        self.visibility_data = self.visibility_data[
            (self.visibility_data['recordingId'] == recording_meta["recordingId"])]

        self.config = config
        self.input_path = config["dataset_dir"]
        self.dataset = config["dataset"].lower()
        self.fixed_blocks_info = fixed_blocks_info
        self.history_length = config["history_length"]

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

        self.tracks = pd.DataFrame(tracks)
        self.tracks_meta = pd.DataFrame(tracks_meta)
        self.recording_meta = recording_meta

        # Check whether tracks and tracks_meta match each other
        error_message = "The tracks file and the tracksMeta file is not matching each other. " \
                        "Please check whether you modified any of these files."
        if len(tracks) != len(tracks_meta):
            logger.error(error_message)
            raise DataError("Failed", error_message)
        for track, track_meta in zip(tracks, tracks_meta):
            if track["trackId"] != track_meta["trackId"]:
                logger.error(error_message)
                raise DataError("Failed", error_message)

        # Determine the first and last frame
        self.minimum_frame = self.tracks_meta["initialFrame"].min()
        self.maximum_frame = self.tracks_meta["finalFrame"].max()
        logger.info("The recording contains tracks from frame {} to {}.", self.minimum_frame, self.maximum_frame)

        # Create a mapping between frame and idxs of tracks for quick lookup during playback
        self.frame_to_track_idxs = {}
        for i_frame in range(self.minimum_frame, self.maximum_frame + 1):
            indices = self.tracks_meta[(self.tracks_meta["initialFrame"] <= i_frame) & (self.tracks_meta["finalFrame"] >= i_frame)].index
            self.frame_to_track_idxs[i_frame] = indices

        # We have to find out timesteps that have atleast one hidden record in the visiblity data.
        # We cannot start from the minimum_frame as we have to include the history as well.
        self.index_map = {}
        for i_frame in range(self.minimum_frame + self.history_length, self.maximum_frame + 1):
            if i_frame % 4 == 0:
                hidden_objects = self.visibility_data[(self.visibility_data['frame'] == i_frame)
                                                      & (self.visibility_data['visibility'] == False)]

                tracks_with_full_history = []
                if len(hidden_objects) > 0:
                    for track_idx in self.frame_to_track_idxs[i_frame]:
                        initial_frame = self.tracks_meta[self.tracks_meta["trackId"] == track_idx]["initialFrame"].item()
                        vehicle_type = self.tracks_meta[self.tracks_meta["trackId"] == track_idx]["class"].item()
                        if initial_frame <= (i_frame - self.history_length) and (vehicle_type == 'car' or vehicle_type == 'truck_bus'):
                            tracks_with_full_history.append(track_idx)

                    tracks_with_limited_visibility = list(hidden_objects["trackId"].drop_duplicates())
                    eligible_tracks = list(set(tracks_with_limited_visibility) & set(tracks_with_full_history))
                    if len(eligible_tracks) == 0: continue
                    self.index_map[i_frame] = random.choice(eligible_tracks)

        # Initialize data variables
        self.track_info_figures = {}

        # Show background image
        background_image_path = self.config["background_image_path"]
        if background_image_path and os.path.exists(background_image_path):
            logger.info("Loading background image from {}", background_image_path)
            self.background_image = cv2.cvtColor(cv2.imread(background_image_path), cv2.COLOR_BGR2RGB)
            (self.image_height, self.image_width) = self.background_image.shape[:2]
        else:
            logger.warning("No background image given or path not valid. Using fallback black background.")
            self.image_height, self.image_width = 1700, 1700
            self.background_image = np.zeros((self.image_height, self.image_width, 3), dtype="uint8")

        # Initialize visualization options
        self.current_frame = self.minimum_frame

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        bb_boxes = []
        self.current_frame = list(self.index_map.keys())[idx]

        ego_vehicle_track_idx = self.index_map[self.current_frame]

        # iterate through all the historical frames upto the current_frame
        historical_obs = []
        for t in range(self.current_frame - self.history_length, self.current_frame):
            map = deepcopy(self.background_image)
            visibility_info = self.visibility_data[(self.visibility_data["trackId"] == ego_vehicle_track_idx)
                                                   & (self.visibility_data["frame"] == t)][["adjacentTrackId", "visibility"]]

            visible_track_ids = visibility_info[(visibility_info["visibility"] == True)]["adjacentTrackId"].drop_duplicates()

            for track_idx in visible_track_ids:
                pts = self._create_polygon(track_idx, t)
                cv2.fillPoly(map, [pts], (0, 255, 0))

            pts = self._create_polygon(ego_vehicle_track_idx, t)
            cv2.fillPoly(map, [pts], (0, 255, 255))

            map_resized = cv2.resize(map, (224, 224), interpolation=cv2.INTER_AREA)

            historical_obs.append(map)

            cv2.imshow('Image with Polygon', map_resized)

        # Extract Ground Truth
        # Find hidden tracks for the selected ego vehicle
        hidden_track_idx = list(self.visibility_data[(self.visibility_data["frame"] == self.current_frame)
                                                   & (self.visibility_data["visibility"] == False)
                                                   & (self.visibility_data["trackId"] == ego_vehicle_track_idx)
                                                   ]["adjacentTrackId"].drop_duplicates())
        hidden_tracks = self.tracks[self.tracks["trackId"].isin(hidden_track_idx)]
        hidden_tracks["existsInFrame"] = hidden_tracks["frame"].apply(lambda x: self.current_frame in x)
        hidden_tracks = hidden_tracks[hidden_tracks["existsInFrame"]]

        pts_ego = self._create_polygon(ego_vehicle_track_idx, self.current_frame)
        cv2.fillPoly(self.background_image, [pts_ego], (0, 255, 0))
        for h in hidden_tracks["trackId"]:
            pts_hidden = self._create_polygon(h, self.current_frame)
            cv2.fillPoly(self.background_image, [pts_hidden], (255, 255, 0))

        cv2.imshow('Image with Polygon', self.background_image)

        return (historical_obs, hidden_tracks)

    def _create_polygon(self, track_idx: int, t: int) -> np.ndarray:
        track = self.tracks[self.tracks["trackId"] == track_idx].iloc[0].to_dict()

        track_meta = self.tracks_meta[self.tracks_meta["trackId"] == track_idx].iloc[0].to_dict()
        initial_frame = track_meta["initialFrame"]
        current_index = t - initial_frame
        if current_index < 0:
            return None

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

        return pts
