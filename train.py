import argparse
import os
from model import MultiScaleResidualNetwork
from dataset import Div2K
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from utils import *


parser = argparse.ArgumentParser(description="Training arguments for MSRN Pytorch")

# Data
parser.add_argument("--data_root", type=str, required=True,
                    help="Root directory of Div2K dataset")
parser.add_argument("--patch_size", type=int, default=128,
                    help="Patch size for training")

# Model
parser.add_argument("--scale", type=int, required=True, options=[2, 3, 4],
                    help="Scale of super-resolution")
parser.add_argument("--residual_blocks", type=int, default=8,
                    help="Number of residual blocks in the network")
parser.add_argument("--residual_channels", type=int, default=64,
                    help="Number of channels at the input and output of residual blocks")

# Platform config
parser.add_argument("--num_workers", type=int, default=os.cpu_count(),
                    help="Number of cpu's used for data loading")
parser.add_argument("--use_cpu", type=bool, default=False,
                    help="Use CPU for training")
parser.add_argument("--num_gpus", type=int, default=1,
                    help="Number of gpu's for training")

# Training specs
parser.add_argument("--epochs", type=int, default=1000,
                    help="Number of epochs for training")
parser.add_argument("--batch_size", type=int, default=16,
                    help="Batch size")
parser.add_argument("--learning_rate", type=int, default=1e-4,
                    help="Learning rate")
parser.add_argument("--beta1", type=float, default=0.9,
                    help="Beta 1 for ADAM")
parser.add_argument("--beta2", type=float, default=0.999,
                    help="Beta 2 for ADAM")
parser.add_argument("--epsilon", type=float, default=1e-8,
                    help="Epsilon for ADAM")

# Loss
parser.add_argument("--loss_fn", type=str, default="L1", options=["L1", "GAN"],
                    help="Loss function to use for training")


if __name__ == "__main__":

    args = parser.parse_args()
    print(args)

    # TODO - loss, training loop
    model = MultiScaleResidualNetwork(scale=args.scale, res_blocks=args.residual_blocks, res_in_features=args.residual_channels, res_out_features=args.residual_channels)
    data_div2k = Div2K(data_root=args.data_root, scale=args.scale, patch_size=args.patch_size)

    optimizer = Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.epsilon)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.5)

    train_sampler, validation_sampler = get_samplers(data_div2k)
    train_loader = DataLoader(data_div2k, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    validation_sampler = DataLoader(data_div2k, batch_size=args.batch_size, sampler=validation_sampler, num_workers=args.num_workers, pin_memory=True)

