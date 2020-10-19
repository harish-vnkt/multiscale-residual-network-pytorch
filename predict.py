import argparse
import os
from model import MultiScaleResidualNetwork
from dataset import *
from torch.utils.data import DataLoader
from utils import *
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging

parser = argparse.ArgumentParser(description="Prediction arguments for MSRN Pytorch")

# Data
parser.add_argument("--data_set", type=str, choices=["Set14"],
                    default="Set14", help="Data set to run prediction on")

# Model
parser.add_argument("--scale", type=int, required=True, choices=[2, 4],
                    help="Scale of super-resolution")
parser.add_argument("--model_file", type=str, required=True,
                    help="Path to .pt file")
parser.add_argument("--results_dir", type=str, required=True,
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


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(filename)s : %(message)s")

    now = datetime.now()
    log_dir = "predict_log-" + now.strftime("%m_%d-%H_%M")

    args = parser.parse_args()
    print(args)

    model = MultiScaleResidualNetwork(scale=args.scale, res_blocks=args.residual_blocks,
                                      res_in_features=args.residual_channels, res_out_features=args.residual_channels)
    logging.debug("Created model")
    data_set = None
    if args.data_set == "Set14":
        data_set = Set14(data_root=args.data_root, scale=args.scale)
    logging.debug("Created dataset")

    prediction_loader = DataLoader(data_set, batch_size=1, num_workers=args.num_workers, pin_memory=True)
    logging.debug("Created prediction loaders")

    if not args.use_cpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    logging.debug("Initialized device : %s", str(device))

    checkpoint = torch.load(args.model_file)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    results_dir = os.path.join(args.results_dir, "x" + str(args.scale))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    writer = SummaryWriter(log_dir)
    logging.debug("Initialized tensorboard directory")

    average_psnr = 0.0
    with torch.no_grad():
        for i, (hr_img, lr_img) in enumerate(prediction_loader):

            image_name = os.path.basename(data_set.hr_image_names[i])
            hr_img_device = hr_img.to(device)
            lr_img_device = lr_img.to(device)

            hr_prediction = model(lr_img_device)

            average_psnr += get_psnr(hr_img_device, hr_prediction)

            write_results(hr_prediction, results_dir, image_name)

    average_psnr /= len(data_set)
    logging.info(f"Average PSNR : {average_psnr}")
