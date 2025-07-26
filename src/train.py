import argparse

from src.dataset.data_loader2 import TrackDataset
from src.models.transformer.graph_weight_encoder import GraphWeightEncoder


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

    return vars(cs.parse_args())


def train(model, data_loader):
    pass


if __name__ == '__main__':
    config = create_args()
    dataset_dir = config["dataset_dir"] + "/"
    track_dataset = TrackDataset(config, dataset_dir)

    model = GraphWeightEncoder().to(config["device"])
    train(model, None)
