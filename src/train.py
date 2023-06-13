import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import torch
import torchvision
from torchvision.transforms import ToTensor, Resize, RandomRotation, RandomPerspective, ColorJitter, RandomApply, \
    GaussianBlur, RandomAdjustSharpness, RandomAutocontrast, RandomEqualize, ToPILImage, RandomVerticalFlip
from lightning.pytorch.tuner import Tuner

import sys
sys.path.append(".")
from src.data.ColorDatamodule import ColorDatamodule
from src.model.SqueezedSqueezeNet import SqueezedSqueezeNet
from src.model.simple_cnn import SimpleCNN
from src.model.nvidia_model import NvidiaModel
from src.model.mobilenet_small import MobileNetSmall
from src.model.SqueezeNet import SqueezeNet
import argparse

from src.model.ColorModule import ColorModule

parser = argparse.ArgumentParser(description='Process model arguments.')
parser.add_argument('--model', type=str,
                    default="SimpleCNN", help='model name')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=1, help='epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--onnx', type=str, default="None", help='onnx')
parser.add_argument('--resolution', type=int, default=64, help='resolution')
parser.add_argument('--precision', type=int, default=16, help='precision')
parser.add_argument('--lr_cycles', type=float, default=1, help='precision')
parser.add_argument('--transformation_probability', type=float, default=0.5, help='precision')
parser.add_argument('--checkpoint', type=str, default="None", help='checkpoint')
parser.add_argument('--test_size', type=float, default=0.2, help='test_size')


def to_onnx(model: ColorModule):
    dummy_input = torch.randn(1, 3, args.resolution, args.resolution)
    torch.onnx.export(model.backbone, dummy_input,
                      f"{args.onnx}.onnx", verbose=True)


def get_model(backbone_name):
    if backbone_name == "SimpleCNN":
        backbone = SimpleCNN(num_classes=3)
    elif backbone_name == "SqueezeNetCustom":
        backbone = SqueezedSqueezeNet(num_classes=3)
    elif backbone_name == "NvidiaModel":
        backbone = NvidiaModel(num_classes=3)
    elif backbone_name == "MobileNetSmall":
        backbone = MobileNetSmall()
    elif backbone_name == "SqueezeNet":
        backbone = SqueezeNet(num_classes=3)
    else:
        raise NotImplementedError(f"Backbone {backbone_name} not implemented")
    return ColorModule(backbone, num_classes=3, lr=args.lr, max_epochs=args.epochs, lr_cycles=args.lr_cycles)

if __name__ == "__main__":
    args = parser.parse_args()
    pl.seed_everything(args.seed)

    train_transformations = torchvision.transforms.Compose(
        [ToTensor(),
         Resize((args.resolution, args.resolution), antialias=True),
         RandomVerticalFlip(p=args.transformation_probability),
         RandomRotation((-30,30)),
         # RandomPerspective(0.05, args.transformation_probability),
         RandomApply([GaussianBlur(3, sigma=(0.05, 0.2))], p=args.transformation_probability),
         RandomAdjustSharpness(0.1, p=args.transformation_probability),
         RandomAutocontrast(p=args.transformation_probability),
        ])
    valid_transformations = torchvision.transforms.Compose(
        [ToTensor(), Resize((args.resolution, args.resolution), antialias=True)])

    data_module = ColorDatamodule("./data/dataset_color", train_transformations,
                                          valid_transformations, batch_size=args.batch_size,
                                  num_workers=6, test_size=args.test_size)
    data_module.setup()
    model = get_model(args.model)
    if args.checkpoint != "None":
        model = model.load_from_checkpoint(args.checkpoint)
    trainer = pl.Trainer(accelerator="auto",
                         precision=args.precision,
                         max_epochs=args.epochs, num_sanity_val_steps=2,
                         logger=WandbLogger(project="ConeColor", name=args.model, config=args),
                         log_every_n_steps=1,
                         callbacks=[
                             ModelCheckpoint(
                                 dirpath="./checkpoints",
                                 filename=args.model+"-{epoch:02d}-{validation_loss:.2f}",
                                 monitor="validation_loss",
                             ),
                             LearningRateMonitor(),
                             EarlyStopping(monitor="validation_loss", patience=3, verbose=True)
                         ],
                         )

    tuner = Tuner(trainer)
    tuner.lr_find(model, datamodule=data_module, min_lr=1e-6, max_lr=1e-1, num_training=1000)

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    if args.onnx != "None":
        to_onnx(model)
