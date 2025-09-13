import argparse
import logging

import torch
import yaml

from src.dataset.data_loader import OGMDataLoader
from src.models.spatio_temporal_encoder import SGATTransformer
import torch.nn as nn

from src.validate import evaluate


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


def train(model, train_data_loader, valid_data_loader, config):
    logging.basicConfig(
        filename=config['model']['log_file'],  # Specify the log file name
        level=logging.INFO,  # Set the logging level (e.g., INFO, DEBUG, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(levelname)s - %(message)s',  # Define the log message format
        filemode='a'  # Set the file mode to 'a' for append, or 'w' for overwrite
    )

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['lr'])
    optimizer.zero_grad()
    loss_fn_aggregated = nn.BCEWithLogitsLoss(reduction='none')

    best_loss = float('inf')

    if config['model']['use_lr_scheduler']:
        lambda1 = lambda epoch: 0.9 ** epoch
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    for epoch in range(config['model']['train_epochs']):  # Example: 10 epochs

        total_loss = 0.0
        total_data_points = 0

        for batch_idx, (inputs, target) in enumerate(train_data_loader):

            # Move data to the correct device
            targets = target.to(config['model']["device"])
            for k, v in inputs.items():
                if k != 'edge_index' and k != 'edge_weights':
                    inputs[k] = v.to(config['model']["device"])

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

            optimizer.zero_grad()
            loss_aggregated.backward()
            optimizer.step()

            if batch_idx % 10 == 0:  # Log every 10 batches
                print(
                    f'Train Epoch {epoch}, Batch {batch_idx}, Loss: {loss_aggregated.item()}, Items: {torch.sum(mask.int())}')
                logging.info(
                    f'Train Epoch {epoch}, Batch {batch_idx}, Loss: {loss_aggregated.item()}, Accuracy: {accuracy.item()}')

                connections = 0
                for edge_weight in inputs['edge_weights']:
                    connections += edge_weight.shape[0]

                print(
                    'Connections: {}, avg nodes: {}'.format(connections, (connections / torch.sum(mask.int())).item()))

        train_loss = total_loss / total_data_points

        print(f'Epoch {epoch}, Training Loss: {train_loss}')
        logging.info(f'Epoch {epoch}, Training Loss: {train_loss}')

        # Validate the model
        valid_loss = evaluate(model, valid_data_loader, config['model']["device"])
        print(f'Epoch {epoch}, Validation Loss: {valid_loss}')
        logging.info(f'Epoch {epoch}, Validation Loss: {valid_loss}')

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_path = config['model']['model_output_path']
            torch.save(model.state_dict(), best_path.format(epoch))
            print(f'New best model saved at epoch {epoch} with validation loss {best_loss}')
            logging.info(f'New best model saved at epoch {epoch} with validation loss {best_loss}')

        if config['model']['use_lr_scheduler']:
            lr_scheduler.step()

    # Save the final model checkpoint
    torch.save(model.state_dict(), config['model']['model_output_path'].format('final'))


if __name__ == '__main__':
    with open("../configs/config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        config['data']['batch_size'] = config['model']['train_batch_size']

    train_dataloader = OGMDataLoader(config['data'], phase='train').create_dataloader()
    valid_dataloader = OGMDataLoader(config['data'], phase='validation').create_dataloader()

    model = SGATTransformer(config['model']).to(config['model']["device"])

    train(model, train_dataloader, valid_dataloader, config)
