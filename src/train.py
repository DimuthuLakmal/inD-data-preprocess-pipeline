import argparse

import torch
import yaml

from src.dataset.data_loader import OGMDataLoader
from src.models.spatio_temporal_encoder import SGATTransformer
from src.models.transformer.graph_weight_encoder import GraphWeightEncoder
import torch.nn as nn


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


def train(model, data_loader, config):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['lr'])
    optimizer.zero_grad()
    loss_fn = nn.BCELoss()

    for epoch in range(config['model']['train_epochs']):  # Example: 10 epochs
        for batch_idx, (inputs, target) in enumerate(data_loader):

            # Move data to the correct device
            targets = target.to(config['model']["device"])
            for k, v in inputs.items():
                if k != 'edge_index' and k != 'edge_weights':
                    inputs[k] = v.to(config['model']["device"])

            mask = inputs['mask']  # Mask indicates the non-padded cells (1: valid, 0: padded)
            outputs = model(inputs, ~mask).squeeze()

            # Calculate the binary cross-entropy loss
            # Masking is applied to ignore unwanted cells
            targets = targets.squeeze() * mask  # Apply mask to targets

            mask_fixed_blocks = (targets != 3)
            targets = targets * mask_fixed_blocks  # Consider cells occupied with fixed blocks as not occupied

            loss = loss_fn(nn.Sigmoid()(outputs), targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:  # Log every 10 batches
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')


if __name__ == '__main__':
    with open("../configs/config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        config['data']['batch_size'] = config['model']['train_batch_size']

    train_dataloader = OGMDataLoader(config['data'], phase='train').create_dataloader()

    model = SGATTransformer(config['model']).to(config['model']["device"])

    train(model, train_dataloader, config)
