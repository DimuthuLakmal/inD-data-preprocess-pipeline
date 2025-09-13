import argparse
import logging

import torch
import torch.nn as nn


def evaluate(model, valid_data_loader, device):
    model.eval()

    loss_fn_aggregated = nn.BCEWithLogitsLoss(reduction='none')

    with torch.no_grad():  # Example: 10 epochs

        total_loss = 0.0
        total_data_points = 0

        for batch_idx, (inputs, target) in enumerate(valid_data_loader):

            # Move data to the correct device
            targets = target.to(device)
            for k, v in inputs.items():
                if k != 'edge_index' and k != 'edge_weights':
                    inputs[k] = v.to(device)

            mask = inputs['mask']  # Mask indicates the non-padded cells (1: valid, 0: padded)
            # Masking is applied to ignore unwanted cells
            mask_fixed_blocks = (targets != 3).squeeze(
                -1)  # Cells occupied with fixed blocks are marked with a 3 in the target
            mask = mask * mask_fixed_blocks  # Consider cells occupied with fixed blocks as not padded

            outputs = model(inputs).squeeze(-1)
            outputs_sig = nn.Sigmoid()(outputs)

            # Calculate the binary cross-entropy loss
            targets = targets.squeeze(-1) * mask
            outputs = outputs * mask  # Apply mask to outputs

            loss_aggregated = loss_fn_aggregated(outputs, targets)
            loss_aggregated = loss_aggregated * mask
            loss_aggregated = loss_aggregated.sum() / (mask.sum().clamp_min(1))
            accuracy = (outputs_sig.round() == targets).float().mean()

            total_loss += loss_aggregated.item()
            total_data_points += (mask.sum().clamp_min(1))

            if batch_idx % 10 == 0:  # Log every 10 batches
                print(
                    f'Batch {batch_idx}, Loss: {loss_aggregated.item()}, Items: {torch.sum(mask.int())}')
                logging.info(
                    f'Batch {batch_idx}, Loss: {loss_aggregated.item()}, Accuracy: {accuracy.item()}')

                connections = 0
                for edge_weight in inputs['edge_weights']:
                    connections += edge_weight.shape[0]

                print(
                    'Connections: {}, avg nodes: {}'.format(connections, (connections / torch.sum(mask.int())).item()))

        valid_loss = total_loss / total_data_points

    return valid_loss
