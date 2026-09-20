import os
import threading

import cv2
import numpy as np
import torch
import grpc

from src.dataset import feature_builder
from src.serving.generated import ogm_inference_pb2, ogm_inference_pb2_grpc
from src.validate import draw_dotted_line

EDGE_KEEP_THRESHOLD = 0.1  # matches the sparse-edge visualization convention in validate.py
EDGE_KEEP_COLOR = (3, 252, 232)
EDGE_DROP_COLOR = (3, 132, 252)
VEHICLE_COLOR = (0, 255, 0)
OCCUPIED_CELL_COLOR = (235, 52, 52)


class OGMInferenceServicer(ogm_inference_pb2_grpc.OGMInferenceServiceServicer):
    """
    Builds the same tensors OGMDataset would build for a frame (via feature_builder,
    shared with training) from a live PredictOccupancyRequest, and runs the model on
    all of the request's occluded cells in a single forward pass.
    """

    def __init__(self, model, background_images, semantic_maps, history_length, device,
                viz_output_dir="../results/serving_visualizations"):
        self.model = model
        self.background_images = background_images
        self.semantic_maps = semantic_maps
        self.history_length = history_length
        self.device = device
        self.viz_output_dir = viz_output_dir
        self._call_counter = 0
        self._counter_lock = threading.Lock()

    def PredictOccupancy(self, request, context):
        with self._counter_lock:
            self._call_counter += 1
            call_id = self._call_counter
        call_dir = os.path.join(self.viz_output_dir, f"call_{call_id:05d}")
        os.makedirs(call_dir, exist_ok=True)

        background_img = self.background_images.get(request.scene_id)
        if background_img is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown scene_id: {request.scene_id}")

        if len(request.vehicles) == 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "At least one vehicle observation is required")
        if len(request.cells) == 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "At least one occluded cell is required")

        expected_t = self.history_length + 1
        for vehicle in request.vehicles:
            if len(vehicle.timesteps) != expected_t:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"vehicle '{vehicle.track_id}' has {len(vehicle.timesteps)} timesteps, "
                    f"expected exactly {expected_t}")

        historical_adjacent_obs = {
            idx: self._build_raw_vehicle_obs(vehicle, expected_t)
            for idx, vehicle in enumerate(request.vehicles)
        }
        last_recorded_t = feature_builder.compute_last_recorded_t(historical_adjacent_obs)

        cells_xy_norm = [
            [cell.cx / (background_img.shape[1] - 1), cell.cy / (background_img.shape[0] - 1)]
            for cell in request.cells
        ]

        edge_weights, edge_index = feature_builder.extract_edge_info(
            historical_adjacent_obs, cells_xy_norm, last_recorded_t)
        historical_adjacent_input, seq_mask = feature_builder.build_vehicle_tensor(
            historical_adjacent_obs, request.scene_id)
        map_obs = self.semantic_maps[request.scene_id]

        inputs = self._to_batch_of_one(
            historical_adjacent_input, seq_mask, cells_xy_norm, map_obs, edge_weights, edge_index)

        self.model.eval()
        with torch.no_grad():
            out_fc, _gates, _l2_loss, _z_mask = self.model(inputs)
        probs = torch.sigmoid(out_fc).squeeze(-1).squeeze(0).cpu().numpy()  # [N_cells]
        z_mask = _z_mask[0]  # batch-of-one: single bipartite graph per call

        response = ogm_inference_pb2.PredictOccupancyResponse()
        for i, (cell, prob) in enumerate(zip(request.cells, probs)):
            is_occupied = bool(prob >= 0.2)
            response.predictions.add(
                cx=cell.cx, cy=cell.cy,
                occupancy_probability=float(prob),
                is_occupied=is_occupied)

            if is_occupied:
                self._draw_cell_edges(call_dir, i, cell, background_img,
                                      historical_adjacent_obs, last_recorded_t,
                                      z_mask, edge_index)
        return response

    @staticmethod
    def _draw_cell_edges(call_dir, cell_idx, cell, background_img, historical_adjacent_obs,
                         last_recorded_t, z_mask, edge_index):
        """Draws, for one predicted-occupied cell, its z-mask-gated connections to every
        adjacent vehicle over the map image - one image per GAT head (collapsing to a single
        un-suffixed image when there's only one head), matching the visualization convention
        of the commented-out debug code in validate.py."""
        edge_src, edge_dst = edge_index
        edges_for_cell = [k for k, dst in enumerate(edge_dst) if dst == cell_idx]

        width, height = background_img.shape[1], background_img.shape[0]
        cell_pt = (int(cell.cx), int(cell.cy))
        num_heads = z_mask.shape[1]

        for h in range(num_heads):
            img = background_img.copy()

            for k in edges_for_cell:
                vehicle_idx = edge_src[k]
                t = last_recorded_t[vehicle_idx]
                if t is None:
                    continue  # vehicle was never actually recorded

                x_norm, y_norm = historical_adjacent_obs[vehicle_idx][t][:2]
                vehicle_pt = (int(x_norm * (width - 1)), int(y_norm * (height - 1)))

                if z_mask[k, h].item() >= EDGE_KEEP_THRESHOLD:
                    cv2.line(img, cell_pt, vehicle_pt, EDGE_KEEP_COLOR, 2)
                else:
                    draw_dotted_line(img, cell_pt, vehicle_pt, EDGE_DROP_COLOR, 2)

                cv2.circle(img, vehicle_pt, 5, VEHICLE_COLOR, -1)

            cv2.circle(img, cell_pt, 5, OCCUPIED_CELL_COLOR, -1)

            filename = f"cell_{cell_idx}.png" if num_heads == 1 else f"cell_{cell_idx}_head{h}.png"
            cv2.imwrite(os.path.join(call_dir, filename), img)

    @staticmethod
    def _build_raw_vehicle_obs(vehicle, expected_t):
        """
        Builds the raw [T, 10] observation array feature_builder expects, matching the
        training-time pickle layout. Column 8 (time_offset = t / history_length, oldest=0,
        current=1) is derived from the timestep's position; column 9 (distance-to-ego) is
        never read by the model, left at 0. A timestep left entirely at its proto default
        (every field == 0) is treated as not-yet-visible/padding, matching training's
        all-zero-row convention.
        """
        history_length = expected_t - 1
        obs = np.zeros((expected_t, 10), dtype=np.float32)
        for t, ts in enumerate(vehicle.timesteps):
            if ts.x_norm == 0 and ts.y_norm == 0 and ts.heading_deg == 0 and \
               ts.x_velocity == 0 and ts.y_velocity == 0 and \
               ts.x_acceleration == 0 and ts.y_acceleration == 0 and \
               ts.vehicle_type == ogm_inference_pb2.CAR:
                continue  # leave this timestep as the zero row already in `obs`

            obs[t] = [ts.x_norm, ts.y_norm, ts.heading_deg,
                     ts.x_velocity, ts.y_velocity,
                     ts.x_acceleration, ts.y_acceleration,
                     float(ts.vehicle_type),
                     t / history_length,
                     0.0]
        return obs

    def _to_batch_of_one(self, historical_adjacent_input, seq_mask, cells_xy_norm, map_obs,
                         edge_weights, edge_index):
        num_vehicles = historical_adjacent_input.shape[0]
        to_tensor = lambda arr, dtype: torch.as_tensor(arr, dtype=dtype, device=self.device).unsqueeze(0)

        return {
            "historical_adjacent_obs": to_tensor(historical_adjacent_input, torch.float32),
            "seq_mask": to_tensor(seq_mask, torch.bool),
            "vehicle_mask": torch.zeros(1, num_vehicles, dtype=torch.bool, device=self.device),
            "hidden_ogm_cells": to_tensor(np.array(cells_xy_norm, dtype=np.float32), torch.float32),
            "map_obs": to_tensor(map_obs.astype(np.float32), torch.float32),
            "edge_weights": [torch.as_tensor(np.expand_dims(np.array(edge_weights, dtype=np.float32), -1),
                                            dtype=torch.float32, device=self.device)],
            "edge_index": [torch.as_tensor(np.array(edge_index, dtype=np.int64),
                                          dtype=torch.int64, device=self.device)],
        }
