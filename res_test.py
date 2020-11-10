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
parser.add_argument('--lr-dir', type=str, required=True)
parser.add_argument('--gt-dir', type=str, required=True)
parser.add_argument('--scale', type=int, default=True)
args = parser.parse_args()

# Using the cuda
cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define the net
model = SRResNet(16, args.scale).to(device)

# Load the param
model.load_state_dict(torch.load(args.weights_file))
model.eval()

test_dataset = SRGanDataset(args.lr_dir, args.gt_dir, in_memory=True, transform=None)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)

# Bicubic
# cv2.imwrite("./data/" + "bicubic.bmp", img)

index = 0
with torch.no_grad():
    for data in test_dataloader:
        index = index + 1
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        bs, c, h, w = inputs.size()
        gt = labels[:, :, : h * args.scale, : w * args.scale]

        output = model(inputs)
        output = output[0].cpu().numpy()
        gt = gt[0].cpu().numpy()

        output = output.transpose(1, 2, 0)
        gt = gt.transpose(1, 2, 0)

        y_output = cv2.cvtColor(output, cv2.COLOR_BGR2YCR_CB)[args.scale:-args.scale, args.scale:-args.scale, :1]
        y_gt = cv2.cvtColor(gt, cv2.COLOR_BGR2YCR_CB)[args.scale:-args.scale, args.scale:-args.scale, :1]

        psnr = img_psnr(y_gt / 255.0, y_output / 255.0)
        print('PSNR: {:.2f}'.format(psnr))

        output = np.array(output * 255.0, ).transpose([1, 2, 0])
        # output = cv2.cvtColor(output, cv2.COLOR_YCR_CB2BGR)
        cv2.imwrite("./data/result/" + "res_%04d.png" % index, output)
