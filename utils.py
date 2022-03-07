import random
import time
import datetime
import sys

from torch.autograd import Variable
import torchvision
import torch
from visdom import Visdom
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


class Logger():
    def __init__(self, n_epochs, batches_epoch, log_interval):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = log_interval
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}
        self.log_int = log_interval

    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write(
            '\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].data[0]
            else:
                self.losses[loss_name] += losses[loss_name].data[0]

            if (i + 1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name] / self.batch * self.log_int))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name] / self.batch * self.log_int))

        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left * self.mean_period / batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title': image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name],
                               opts={'title': image_name})

        # End of epoch
        if (self.batch > self.batches_epoch):
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]),
                                                                 Y=np.array([loss / self.batch]),
                                                                 opts={'xlabel': 'epochs', 'ylabel': loss_name,
                                                                       'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss / self.batch]),
                                  win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = self.log_int
            sys.stdout.write('\n')
        else:
            self.batch += self.log_int


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
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
