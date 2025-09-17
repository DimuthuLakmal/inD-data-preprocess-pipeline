import argparse
import logging

import torch
import torch.nn as nn
import cv2


def evaluate(model, valid_data_loader, device):
    model.eval()

    loss_fn = nn.BCELoss(reduce=False)

    with torch.no_grad():  # Example: 10 epochs

        total_loss = 0.0
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

            seq_mask = inputs['seq_mask']  # Sequence mask for the historical observations
            vehicle_mask = inputs['vehicle_mask']
            cell_feat = inputs['hidden_ogm_cells']
            veh_feat = inputs['historical_adjacent_obs']
            map = inputs['map_obs']
            outputs = model(veh_feat, cell_feat, seq_mask, mask.unsqueeze(-1), vehicle_mask, map).squeeze()
            outputs_sig = nn.Sigmoid()(outputs)

            # Calculate the binary cross-entropy loss
            targets = targets.squeeze() * mask
            outputs = outputs * mask  # Apply mask to outputs
            outputs_sig = outputs_sig * mask  # Apply mask to outputs

            loss_aggregated = loss_fn(outputs_sig, targets)
            loss_aggregated = loss_aggregated * mask
            scene_wise_loss = list(loss_aggregated.detach().cpu().numpy())

            loss_aggregated = loss_aggregated.sum() / (mask.sum().clamp_min(1))
            accuracy = (outputs_sig.round() == targets).float().mean()

            total_loss += loss_aggregated.item()
            batch_itr += 1

            # print(
            #     f'Batch {batch_idx}, Loss: {loss_aggregated.item()}, Items: {torch.sum(mask.int())}, Scenes: {scene_data}')
            # # print(f'Batch {batch_idx}, Loss: {loss_aggregated.item()}')
            logging.info(f'Batch {batch_idx}, Loss: {loss_aggregated.item()}, Accuracy: {accuracy.item()}')

        valid_loss = total_loss / batch_itr

    return valid_loss
