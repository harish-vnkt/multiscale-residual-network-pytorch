import argparse
import os
from model import MultiScaleResidualNetwork
from dataset import Div2K
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from utils import *
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging


parser = argparse.ArgumentParser(description="Training arguments for MSRN Pytorch")

# Data
parser.add_argument("--data_root", type=str, required=True,
                    help="Root directory of Div2K dataset")
parser.add_argument("--patch_size", type=int, default=128,
                    help="Patch size for training")

# Model
parser.add_argument("--scale", type=int, required=True, choices=[2, 3, 4],
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
parser.add_argument("--loss_fn", type=str, default="L1", choices=["L1", "GAN"],
                    help="Loss function to use for training")

# Save and display
parser.add_argument("--resume", type=bool, default=False,
                    help="If training needs to be resumed")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoint",
                    help="Path to checkpoint directory")
parser.add_argument("--checkpoint_file", type=str, default="last.pt",
                    help="Path to checkpoint file if training needs to be resumed")
parser.add_argument("--results_dir", type=str, default="results",
                    help="Path to save trained models")


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(filename)s : %(message)s")

    now = datetime.now()
    log_dir = "log-" + now.strftime("%m_%d-%H_%M")

    args = parser.parse_args()
    print(args)

    model = MultiScaleResidualNetwork(scale=args.scale, res_blocks=args.residual_blocks, res_in_features=args.residual_channels, res_out_features=args.residual_channels)
    logging.debug("Created model")
    data_div2k = Div2K(data_root=args.data_root, scale=args.scale, patch_size=args.patch_size)
    logging.debug("Created dataset")

    optimizer = Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.epsilon)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.5)
    logging.debug("Created optimizer and scheduler")

    train_sampler, validation_sampler = get_samplers(data_div2k)
    train_loader = DataLoader(data_div2k, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    validation_loader = DataLoader(data_div2k, batch_size=1, sampler=validation_sampler, num_workers=args.num_workers, pin_memory=True)
    logging.debug("Created validation and test loaders")

    if args.loss_fn == 'L1':
        loss = nn.L1Loss()

    if not args.use_cpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    logging.debug("Initialized device : %s", str(device))

    start_epoch = 0

    checkpoint_file = os.path.join(args.checkpoint_dir, args.checkpoint_file)
    if args.resume:
        logging.debug("Resuming training")
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model'])
        logging.debug("Loaded model")
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.debug("Loaded optimizer")
        scheduler.load_state_dict(checkpoint['scheduler'])
        logging.debug("Loaded scheduler")
        start_epoch = checkpoint['epoch'] + 1
        logging.info("Starting from epoch %d", start_epoch)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    results_dir_scale = os.path.join(args.results_dir, "X" + str(args.scale))
    if not os.path.exists(results_dir_scale):
        os.makedirs(results_dir_scale)

    model.to(device)
    writer = SummaryWriter(log_dir)
    logging.debug("Initialized tensorboard directory")

    for epoch in range(start_epoch, args.epochs):

        model_name = os.path.join(results_dir_scale, "epoch" + str(epoch) + ".pt")
        model.train()
        epoch_loss = 0.0

        for batch, (hr_patch, lr_patch) in enumerate(train_loader):

            hr_patch_device = hr_patch.to(device)
            lr_patch_device = lr_patch.to(device)

            hr_prediction = model(lr_patch_device)
            batch_loss = loss(hr_prediction, hr_patch_device)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()

            logging.info(f"Epoch {epoch + 1} : Samples[{(batch + 1) * args.batch_size}/{len(train_loader) * args.batch_size}] Average batch loss : {epoch_loss / (batch + 1)}")

        scheduler.step()
        writer.add_scalar("Epoch_loss", epoch_loss, epoch + 1)
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch}, checkpoint_file)
        logging.debug("Saved checkpoint")

        if epoch % 100 == 99 or epoch == args.epochs - 1:
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch}, model_name)
            logging.debug(f"Saved model at {model_name}")

        model.eval()
        epoch_psnr = 0.0

        logging.debug(f"Validating epoch {epoch + 1}")
        with torch.no_grad():
            for hr_patch, lr_patch in validation_loader:

                hr_patch_device = hr_patch.to(device)
                lr_patch_device = lr_patch.to(device)

                hr_prediction = model(lr_patch_device)

                epoch_psnr += get_psnr(hr_patch_device, hr_prediction)

        epoch_psnr /= len(validation_loader)
        logging.info(f"Epoch {epoch + 1} : Average validation PSNR : {epoch_psnr}")
        writer.add_scalar("PSNR", epoch_psnr, epoch + 1)
