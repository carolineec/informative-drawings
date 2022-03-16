import random
import time
import datetime
import sys

from torch.autograd import Variable
import torchvision
import torch
import numpy as np

def createNRandompatches(img1, img2, N, patch_size, clipsize=224):
    myw = img1.size()[2]
    myh = img1.size()[3]

    patches1 = []
    patches2 = []

    for i in range(N):
        xcoord = int(torch.randint(myw - patch_size, ()))
        ycoord = int(torch.randint(myh - patch_size, ()))
        patch1 = img1[:, :, xcoord:xcoord+patch_size, ycoord:ycoord+patch_size]
        patches1 += [torch.nn.functional.interpolate(patch1, size=clipsize)]
        patch2 = img2[:, :, xcoord:xcoord+patch_size, ycoord:ycoord+patch_size]
        patches2 += [torch.nn.functional.interpolate(patch2, size=clipsize)]

    return patches1, patches2

def tensor2image(tensor):
    image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)

def channel2width(geom):
    num_chan = int(geom.size()[1])
    chan = 0
    imgs = []
    while chan < num_chan:
        grabme = geom[:, chan:chan+3, :, :]
        imgs += [grabme]
        chan += 3
    input_img_fake = torch.cat(imgs, dim=3)
    return input_img_fake

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []
        self.cond = []

    def push_and_pop(self, data):
        to_return = []
        to_return_cond = []
        for idx, element in enumerate(data[0].data):
            e_cond = data[1].data[idx]

            e_cond = torch.unsqueeze(e_cond, 0)
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)

                self.cond.append(e_cond)
                to_return_cond.append(e_cond)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element

                    to_return_cond.append(self.cond[i].clone())
                    self.cond[i] = e_cond
                else:
                    to_return.append(element)

                    to_return_cond.append(e_cond)

        return Variable(torch.cat(to_return)), Variable(torch.cat(to_return_cond))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
