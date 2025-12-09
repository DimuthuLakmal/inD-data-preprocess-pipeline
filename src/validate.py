import argparse
import logging
import os

import numpy as np

import torch
import torch.nn as nn
import cv2
from fvcore.nn import FlopCountAnalysis

from src.utils.metrics import compute_metrics


def evaluate(model, valid_data_loader, device, writer=None, epoch=0):
    # Loading background images
    background_images = {}
    for scene_id in [0, 7, 8, 18, 19]:
        # Store background images for scenes
        bg_path = os.path.join("../data/", 'maps', f"{scene_id}_background.png")
        img = cv2.imread(bg_path)
        background_images[int(scene_id)] = img

    model.eval()

    loss_fn_aggregated = nn.BCEWithLogitsLoss(reduction='none')

    with torch.no_grad():  # Example: 10 epochs

        total_loss = 0.0
        v_total = {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        v_batches = 0
        v_roc = []
        batch_itr = 0

        for batch_idx, (inputs, target) in enumerate(valid_data_loader):

            # Move data to the correct device
            targets = target.to(device)
            for k, v in inputs.items():
                if k != 'edge_index' and k != 'edge_weights':
                    inputs[k] = v.to(device)

            mask = inputs['mask'].squeeze()  # Mask indicates the non-padded cells (1: valid, 0: padded)
            # Masking is applied to ignore unwanted cells
            mask_fixed_blocks = (
                    targets != 3).squeeze()  # Cells occupied with fixed blocks are marked with a 3 in the target
            mask = mask * mask_fixed_blocks  # Consider cells occupied with fixed blocks as not padded

            outputs, gates, l2_loss, z_masks = model(inputs)
            outputs = outputs.squeeze()
            outputs_sig = nn.Sigmoid()(outputs)

            flops = FlopCountAnalysis(model, inputs)
            print("Total FLOPs: ", flops.total())

            # Calculate the binary cross-entropy loss
            targets = targets.squeeze() * mask
            outputs = outputs * mask  # Apply mask to outputs
            outputs_sig = outputs_sig * mask  # Apply mask to outputs

            loss_aggregated = loss_fn_aggregated(outputs, targets)
            loss_aggregated = loss_aggregated * mask
            scene_wise_loss = list(loss_aggregated.detach().cpu().numpy())
            loss_avg = loss_aggregated.sum() / (mask.sum().clamp_min(1))

            v_total["loss"] += loss_avg.item()
            m = compute_metrics(outputs_sig, targets, mask)
            for k in ("accuracy", "precision", "recall", "f1"):
                v_total[k] += m[k]
            if m["roc_auc"] is not None:
                v_roc.append(m["roc_auc"])

            batch_itr += 1

            scene_data = [(arr[0], scene_loss) for (arr, scene_loss) in zip(list(inputs["scene_id"].detach().cpu().numpy()), scene_wise_loss)]

            # Check sparse edge cases
            # cell_obs = inputs['hidden_ogm_cells']
            # v_obs = inputs['historical_adjacent_obs']
            #
            # preds = (outputs_sig >= 0.5).float()
            #
            # for b_i in range(cell_obs.shape[0]):
            #     map_img = background_images[inputs['scene_id'][b_i].item()]
            #
            #     x_coords = v_obs[b_i, :, :, 0:1] * map_img.shape[1]
            #     y_coords = v_obs[b_i, :, :, 1:2] * map_img.shape[0]
            #
            #     cell_x_coords = (cell_obs[b_i, :, 0:1] * map_img.shape[1]).squeeze()
            #     cell_y_coords = (cell_obs[b_i, :, 1:2] * map_img.shape[0]).squeeze()
            #
            #     if preds[b_i].item() == targets[b_i].item():
            #         z_masks[b_i] = z_masks[b_i].permute(1, 0)
            #         for h, z_mask in enumerate(z_masks[b_i]):
            #             edge_val = z_mask.detach().cpu().numpy()
            #             cell_pts = (int(cell_x_coords.item()), int(cell_y_coords.item()))
            #
            #             map_img_t = map_img.copy()
            #             for j in range(edge_val.shape[0]):
            #                 pt = (int(x_coords[j, -1, 0].item()), int(y_coords[j, -1, 0].item()))
            #                 for t_ext in range(5):
            #                     if pt[0] == 0 and pt[1] == 0:
            #                         pt = (int(x_coords[j, -1 - t_ext, 0].item()), int(y_coords[j, -1 - t_ext, 0].item()))
            #
            #                 if pt[0] == 0 and pt[1] == 0:
            #                     continue
            #
            #                 if edge_val[j] >= 0.1:
            #                     # cv2.line(map_img_t, cell_pts, pt, lerp_color(edge_val[j]), 2)
            #                     cv2.line(map_img_t, cell_pts, pt, (3, 252, 232), 2)
            #                 else:
            #                     draw_dotted_line(map_img_t, cell_pts, pt, (3, 132, 252), 2)
            #
            #
            #                 cv2.circle(map_img_t, pt, 5, (0, 255, 0), -1)
            #
            #             if targets[b_i].item() == 1:
            #                 cv2.circle(map_img_t, cell_pts, 5, (235, 52, 52), -1)
            #             else:
            #                 cv2.circle(map_img_t, cell_pts, 5, (210, 3, 252), -1)
            #
            #             cv2.imwrite(f'../results/edge_masks/{batch_idx}_{b_i}_{h}.png', map_img_t)


    valid_loss = v_total["loss"] / batch_itr
    writer.add_scalar("val/loss_epoch", valid_loss, epoch)
    writer.add_scalar("val/accuracy_epoch", v_total["accuracy"] / batch_itr, epoch)
    writer.add_scalar("val/precision_epoch", v_total["precision"] / batch_itr, epoch)
    writer.add_scalar("val/recall_epoch", v_total["recall"] / batch_itr, epoch)
    writer.add_scalar("val/f1_epoch", v_total["f1"] / batch_itr, epoch)
    if len(v_roc) > 0:
        writer.add_scalar("val/roc_auc_epoch", float(np.mean(v_roc)), epoch)

    print(f'Epoch {epoch}, Validation Loss: {valid_loss}')
    logging.info(f'Epoch {epoch}, Validation Loss: {valid_loss}')

    return valid_loss

def draw_dotted_line(img, pt1, pt2, color, thickness=1, dot_length=5, gap_length=5):
    """
    Draws a dotted line on an OpenCV image.

    Args:
        img (numpy.ndarray): The image on which to draw the line.
        pt1 (tuple): The starting point (x, y) of the line.
        pt2 (tuple): The ending point (x, y) of the line.
        color (tuple): The color of the line in BGR format (e.g., (255, 0, 0) for blue).
        thickness (int): The thickness of the dots.
        dot_length (int): The length of each individual dot segment.
        gap_length (int): The length of the gap between dots.
    """
    dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    num_segments = int(dist / (dot_length + gap_length))

    for i in range(num_segments):
        start_ratio = i * (dot_length + gap_length) / dist
        end_ratio = (i * (dot_length + gap_length) + dot_length) / dist

        if end_ratio > 1:  # Prevent drawing beyond the line's end
            end_ratio = 1

        # Calculate start and end points for the current dot segment
        dot_start_x = int(pt1[0] * (1 - start_ratio) + pt2[0] * start_ratio)
        dot_start_y = int(pt1[1] * (1 - start_ratio) + pt2[1] * start_ratio)
        dot_end_x = int(pt1[0] * (1 - end_ratio) + pt2[0] * end_ratio)
        dot_end_y = int(pt1[1] * (1 - end_ratio) + pt2[1] * end_ratio)

        cv2.line(img, (dot_start_x, dot_start_y), (dot_end_x, dot_end_y), color, thickness)


def lerp_color(v):
    """
    v ∈ [0,1] and returns a color between Yellow → Orange in BGR format.
    """
    # Yellow RGB: (255, 255, 0)
    # Orange RGB: (255, 165, 0)

    yellow = np.array([255, 255, 0], dtype=np.float32)
    orange = np.array([255, 0, 0], dtype=np.float32)

    rgb = yellow * (1 - v) + orange * v  # Linear interpolation
    bgr = rgb[::-1]  # Convert RGB → BGR for OpenCV

    return tuple(int(c) for c in bgr)
