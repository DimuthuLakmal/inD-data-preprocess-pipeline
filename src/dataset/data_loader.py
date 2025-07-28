from torch.utils.data import DataLoader

from src.dataset.dataset import OGMDataset


class OGMDataLoader():
    def __init__(self, cfg, phase='train'):
        self.cfg = cfg
        self.phase = phase

    def create_dataloader(self):
        dataset = OGMDataset(self.cfg)
        for i, sample in enumerate(dataset):
            print(i, sample)

        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.cfg.TRAIN.BATCH_SIZE if self.phase == 'train' else self.cfg.TEST.BATCH_SIZE,
                                shuffle=(self.phase == 'train'),
                                num_workers=self.cfg.TRAIN.WORKERS_NUM)
        return dataloader