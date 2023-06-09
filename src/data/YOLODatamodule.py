# # create datamodule
# import torch
# import lightning as pl
# from src.data.dataset import LineFollowingDataset
#
#
# class LineFollowingDataModule(pl.LightningDataModule):
#     def __init__(self, root_folder, train_run_ids, valid_run_ids, train_transformations, valid_transformations,
#                  batch_size=32, num_workers=0):
#         super().__init__()
#         self.root_folder = root_folder
#         self.train_run_ids = train_run_ids
#         self.valid_run_ids = valid_run_ids
#         self.train_transformations = train_transformations
#         self.valid_transformations = valid_transformations
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#
#     def train_dataloader(self):
#         return torch.utils.data.DataLoader(
#             LineFollowingDataset(self.root_folder, self.train_run_ids, self.train_transformations),
#             batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
#
#     def val_dataloader(self):
#         return torch.utils.data.DataLoader(
#             LineFollowingDataset(self.root_folder, self.valid_run_ids, self.valid_transformations),
#             batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
#
#     def test_dataloader(self):
#         return torch.utils.data.DataLoader(
#             LineFollowingDataset(self.root_folder, self.valid_run_ids, self.valid_transformations),
#             batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
#
#     def get_mean_and_std(self, set: str = 'train'):
#         if set == 'train':
#             dataloader = self.train_dataloader()
#         elif set == 'val':
#             dataloader = self.val_dataloader()
#         elif set == 'test':
#             dataloader = self.test_dataloader()
#         channels_sum, channels_squared_sum, num_batches = 0, 0, 0
#         for data, _ in dataloader:
#             # Mean over batch, height and width, but not over the channels
#             channels_sum += torch.mean(data, dim=[0, 2, 3])
#             channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
#             num_batches += 1
#
#         mean = channels_sum / num_batches
#
#         # std = sqrt(E[X^2] - (E[X])^2)
#         std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
#
#         return mean, std
