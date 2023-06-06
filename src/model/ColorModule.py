import torchmetrics
from torch import nn
import torch
import lightning as pl
from src.model.simple_cnn import SimpleCNN


class ColorModule(pl.pytorch.LightningModule):
    def __init__(self, backbone=None, num_classes=2, lr=1e-3, max_epochs=20, lr_cycles=1):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.max_epochs = max_epochs
        self.lr_cycles = lr_cycles

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.criterion(y_hat, y)
        # print("====================================================+")
        # print(nn.Softmax(dim=1)(y_hat))
        # print(y)
        self.log('train_loss', loss)
        self.train_acc(y_hat, y.argmax(dim=1))
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.criterion(y_hat, y)
        self.log('validation_loss', loss)
        self.valid_acc(y_hat, y.argmax(dim=1))
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        self.test_acc(y_hat, y.argmax(dim=1))
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,  int(self.max_epochs//self.lr_cycles))
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }
