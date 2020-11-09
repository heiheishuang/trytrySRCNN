import argparse
import torch
import copy
import os
from torch import nn
from tqdm import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

from models import SRResNet
from utils import AverageMeter, img_psnr
from load_data import SRGanTrainDataset, SRGanTestDataset, Crop

# Init the param
parser = argparse.ArgumentParser()
parser.add_argument('--lr-file', type=str, required=True)
parser.add_argument('--gt-file', type=str, required=True)
parser.add_argument('--test-lr-file', type=str, required=True)
parser.add_argument('--test-gt-file', type=str, required=True)
parser.add_argument('--outputs-dir', type=str, required=True)
parser.add_argument("--patch-size", type=int, default=33)
parser.add_argument('--scale', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--num-epochs', type=int, default=400)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args()

torch.manual_seed(666)
if not os.path.exists(args.outputs_dir):
    os.makedirs(args.outputs_dir)

# Load the dataset
transform = transforms.Compose([Crop(args.scale, args.patch_size)])
train_dataset = SRGanTrainDataset(args.lr_file, args.gt_file, in_memory=True, transform=transform)
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

# test_dataset = SRGanTestDataset(args.eval_file)
# test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)
test_dataset = SRGanTrainDataset(args.test_lr_file, args.test_gt_file)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)

# Use in the cuda
cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Defined the network
net = SRResNet(16, 4).to(device)
optimizer = optim.Adam(net.parameters())

# Define the loss func
mse_loss = nn.MSELoss()

# Train
best_weights = copy.deepcopy(net.state_dict())
best_psnr = 0.0
best_epoch = 0
for epoch in range(args.num_epochs):
    net.train()
    epoch_losses = AverageMeter()

    with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
        t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

        for data in train_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = net(inputs)

            print(preds.shape)
            print(labels.shape)
            loss = mse_loss(preds, labels)

            epoch_losses.update(loss.item(), len(inputs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            t.update(len(inputs))

    torch.save(net.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

    net.eval()
    epoch_psnr = AverageMeter()

    for data in test_dataloader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = net(inputs).clamp(0.0, 1.0)

        loss = img_psnr(preds, labels)
        epoch_psnr.update(loss, len(inputs))

    print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

    if epoch_psnr.avg > best_psnr:
        best_epoch = epoch
        best_psnr = epoch_psnr.avg
        best_weights = copy.deepcopy(net.state_dict())

print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
