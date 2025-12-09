import argparse

import torch
import yaml

from dataset.data_loader import OGMDataLoader
from models.spatio_temporal_encoder import SGATTransformer
from validate import evaluate

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


if __name__ == '__main__':
    with open("../configs/config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        config['data']['batch_size'] = config['model']['test_batch_size']

    valid_dataloader = OGMDataLoader(config['data'], phase='validation').create_dataloader()

    model = SGATTransformer(config['model']).to(config['model']["device"])
    model.load_state_dict(torch.load(config['model']['model_output_path'].format(133)))
    model = model.to(config['model']["device"])

    test_loss = evaluate(model, valid_dataloader, config['model']["device"])
    print(f'Validation loss {test_loss}')