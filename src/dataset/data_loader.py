from torch.utils.data import DataLoader

from src.dataset.collate_fn import custom_collate
from src.dataset.dataset import OGMDataset


class OGMDataLoader():
    def __init__(self, cfg, phase='train'):
        self.cfg = cfg
        self.phase = phase

    def create_dataloader(self, phase=None):
        dataset = OGMDataset(self.cfg, self.phase)

        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.cfg['batch_size'],
                                shuffle=(self.phase == 'train'),
                                collate_fn=custom_collate,
                                num_workers=self.cfg['num_workers'])
        return dataloader