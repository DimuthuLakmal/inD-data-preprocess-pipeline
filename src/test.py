import argparse

import torch
import yaml

from dataset.data_loader import OGMDataLoader
from models.transformer.spatial_temporal_encoder_decoder import CellsFromVehicles, CellFromVehicleAndMap
import torch.nn as nn

from utils.histogram import plot_histogram


def create_args():
    cs = argparse.ArgumentParser(description="Dataset Tracks Visualizer")
    # --- Input ---
    cs.add_argument('--dataset_dir', default="../data/",
                    help="Path to directory that contains the dataset csv files.", type=str)
    cs.add_argument('--visualizer_params_dir', default="../data/visualizer_params/",
                    help="Name of the recording given by a number with a leading zero.", type=str)
    cs.add_argument('--history_length', default="20",
                    help="Number of previous timesetps that includes in the historical observations of a data entry",
                    type=int)
    cs.add_argument('--dataset', default="ind",
                    help="The dataset to use for training and testing",
                    type=str)

    return vars(cs.parse_args())


def test(model, data_loader, config):
    model.eval()

    loss_fn = nn.BCELoss(reduce=False)
    loss_fn_aggregated = nn.BCEWithLogitsLoss(reduction='none')

    first_batch = True

    for epoch in range(config['model']['train_epochs']):  # Example: 10 epochs
        for batch_idx, (inputs, target) in enumerate(data_loader):

            # Move data to the correct device
            targets = target.to(config['model']["device"])
            for k, v in inputs.items():
                if k != 'edge_index' and k != 'edge_weights':
                    inputs[k] = v.to(config['model']["device"])

            mask = inputs['mask']  # Mask indicates the non-padded cells (1: valid, 0: padded)
            # Masking is applied to ignore unwanted cells
            mask_fixed_blocks = (targets != 3).squeeze()  # Cells occupied with fixed blocks are marked with a 3 in the target
            mask = mask * mask_fixed_blocks  # Consider cells occupied with fixed blocks as not padded

            seq_mask = inputs['seq_mask']  # Sequence mask for the historical observations
            vehicle_mask = inputs['vehicle_mask']
            cell_feat = inputs['hidden_ogm_cells']
            veh_feat = inputs['historical_adjacent_obs']
            map = inputs['map_obs']
            outputs = model(veh_feat, cell_feat, seq_mask, mask, vehicle_mask, map).squeeze()
            outputs_sig = nn.Sigmoid()(outputs)

            # Calculate the binary cross-entropy loss
            targets = targets.squeeze() * mask
            outputs = outputs * mask  # Apply mask to outputs
            outputs_sig = outputs_sig * mask  # Apply mask to outputs

            loss_aggregated = loss_fn_aggregated(outputs, targets)
            loss_aggregated = loss_aggregated * mask
            loss_aggregated = loss_aggregated.sum() / (mask.sum().clamp_min(1))
            loss = loss_fn(outputs_sig.view(-1), targets.view(-1))

            if batch_idx % 10 == 0:  # Log every 10 batches
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss_aggregated.item()}, Items: {torch.sum(mask.int())}')

                connections = 0
                for edge_weight in inputs['edge_weights']:
                    connections += edge_weight.shape[0]

                print('Connections: {}, avg nodes: {}'.format(connections, (connections/torch.sum(mask.int())).item()))


if __name__ == '__main__':
    with open("../configs/config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        config['data']['batch_size'] = config['model']['train_batch_size']

    train_dataloader = OGMDataLoader(config['data'], phase='train').create_dataloader()

    model = CellFromVehicleAndMap()
    model.load_state_dict(torch.load(config['model']['model_output_path']))
    model = model.to(config['model']["device"])

    test(model, train_dataloader, config)
