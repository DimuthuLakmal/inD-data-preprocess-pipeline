import argparse
import logging
import numpy as np

import torch
import torch.nn as nn
import cv2

from src.utils.metrics import compute_metrics


def evaluate(model, valid_data_loader, device, writer=None, epoch=0):
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

            outputs, gates, l2_loss = model(inputs)
            outputs = outputs.squeeze()
            outputs_sig = nn.Sigmoid()(outputs)

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
