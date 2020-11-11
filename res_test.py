import argparse

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from torch.utils.data.dataloader import DataLoader
from load_data import SRGanDataset
from models import SRResNet
from utils import img_psnr

# Define the param
parser = argparse.ArgumentParser()
parser.add_argument('--weights-file', type=str, required=True)
parser.add_argument('--lr-file', type=str, required=True)
parser.add_argument('--gt-file', type=str, required=True)
parser.add_argument('--scale', type=int, default=True)
args = parser.parse_args()

# Using the cuda
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the param
dataset = SRGanDataset(gt_path=args.gt_file, lr_path=args.lr_file, in_memory=False, transform=None)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

# Define the net
model = SRResNet(16, args.scale)
model.load_state_dict(torch.load(args.weights_file))
model = model.to(device)
model.eval()

index = 0
with torch.no_grad():
    for data in loader:
        index = index + 1
        lr, gt = data

        lr = lr.to(device)
        gt = gt.to(device)

        _, _, height, weight = lr.size()
        gt = gt[:, :, : height * args.scale, : weight * args.scale]

        output = model(lr)

        output = output[0].cpu().numpy()

        output[output > 1.0] = 1.0
        output[output < 0.0] = 0.0
        gt = gt[0].cpu().numpy()

        output = output.transpose(1, 2, 0)
        gt = gt.transpose(1, 2, 0)

        y_out = cv2.cvtColor(output, cv2.COLOR_RGB2YCR_CB)
        y_out = y_out[args.scale:-args.scale, args.scale:-args.scale, :1]
        print(y_out.shape)

        y_gt = cv2.cvtColor(gt, cv2.COLOR_RGB2YCR_CB)
        y_gt = y_gt[args.scale:-args.scale, args.scale:-args.scale, :1]

        y_out = torch.from_numpy(y_out).to(device)
        y_gt = torch.from_numpy(y_gt).to(device)

        psnr = img_psnr(y_out / 255.0, y_gt / 255.0)

        print('psnr : %04f \n' % psnr)

        result = cv2.cvtColor(output * 255.0, cv2.COLOR_RGB2BGR).astype(np.uint8)
        cv2.imwrite('./data/res_%04d.png' % index, result)
