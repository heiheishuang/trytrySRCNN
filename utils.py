import torch


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count = self.count + n
        self.avg = self.sum / self.count


def img_psnr(img1, img2):
    return 10 * torch.log10(1.0 / torch.mean((img1 - img2) ** 2))
