#!/usr/bin/python3

import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from model import Generator, GlobalGenerator2, InceptionV3
# from utils import ReplayBuffer
from utils import LambdaLR
from utils import channel2width
from utils import weights_init_normal
from utils import createNRandompatches
from dataset import UnpairedDepthDataset
import utils_pl
from collections import OrderedDict
import util.util as util
import networks
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help='name of this experiment')
parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='Where checkpoints are saved')
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
parser.add_argument('--cuda', action='store_true', help='use GPU computation', default=True)
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')

###loading data
parser.add_argument('--dataroot', type=str, default='datasets/vangogh2photo/', help='photograph directory root directory')
parser.add_argument('--root2', type=str, default='', help='line drawings dataset root directory')
parser.add_argument('--depthroot', type=str, default='', help='dataset of corresponding ground truth depth maps')
parser.add_argument('--feats2Geom_path', type=str, default='checkpoints/feats2Geom/feats2depth.pth', 
                                help='path to pretrained features to depth map network')

### architecture and optimizers
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for optimizer')
parser.add_argument('--decay_epoch', type=int, default=50, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--geom_nc', type=int, default=3, help='number of channels of geom data')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
parser.add_argument('--netD', type=str, default='basic', help='selects model to use for netD')
parser.add_argument('--n_blocks', type=int, default=3, help='number of resnet blocks for generator')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
parser.add_argument('--disc_sigmoid', type=int, default=0, help='use sigmoid in disc loss')
parser.add_argument('--every_feat', type=int, default=1, help='use transfer features for recog loss')
parser.add_argument('--finetune_netGeom', type=int, default=1, help='make geometry networks trainable')

### loading from checkpoints
parser.add_argument('--load_pretrain', type=str, default='', help='where to load file if wanted')
parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load from if continue_train')

### dataset options
parser.add_argument('--mode', type=str, default='train', help='train, val, test, etc')
parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

######## loss weights
parser.add_argument("--cond_cycle", type=float, default=0.1, help="weight of the appearance reconstruction loss")
parser.add_argument("--condGAN", type=float, default=1.0, help="weight of the adversarial style loss")
parser.add_argument("--cond_recog", type=float, default=10.0, help="weight of the semantic loss")
parser.add_argument("--condGeom", type=float, default=10.0, help="weight of the geometry style loss")

### geometry loss options
parser.add_argument('--use_geom', type=int, default=1, help='include the geometry loss')
parser.add_argument('--midas', type=int, default=1, help='use midas depth map')

### semantic loss options
parser.add_argument('--N_patches', type=int, default=1, help='number of patches for clip')
parser.add_argument('--patch_size', type=int, default=128, help='patchsize for clip')
parser.add_argument('--num_classes', type=int, default=55, help='number of classes for inception')
parser.add_argument('--cos_clip', type=int, default=0, help='use cosine similarity for CLIP semantic loss')

### save options
parser.add_argument('--save_epoch_freq', type=int, default=1000, help='how often to save the latest model in steps')
parser.add_argument('--slow', type=int, default=0, help='only frequently save netG_A, netGeom')
parser.add_argument('--log_int', type=int, default=50, help='display frequency for tensorboard')


opt = parser.parse_args()
print(opt)

checkpoints_dir = opt.checkpoints_dir 
name = opt.name

from util.visualizer2 import Visualizer
tensor2im = util.tensor2imv2


visualizer = Visualizer(checkpoints_dir, name, tf_log=True, isTrain=True)
print('------------------- created visualizer -------------------')

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks

netG_A = 0
netG_A = Generator(opt.input_nc, opt.output_nc, opt.n_blocks)
netG_B = Generator(opt.output_nc, opt.input_nc, opt.n_blocks)

if opt.use_geom == 1:

    netGeom = GlobalGenerator2(768, opt.geom_nc, n_downsampling=1, n_UPsampling=3)
    netGeom.load_state_dict(torch.load(opt.feats2Geom_path))
    print("Loading pretrained features to depth network from %s"%opt.feats2Geom_path)
    if opt.finetune_netGeom == 0:
        netGeom.eval()
else:
    opt.finetune_netGeom = 0


D_input_nc_B = opt.output_nc
D_input_nc_A = opt.input_nc

netD_B = networks.define_D(D_input_nc_B, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, use_sigmoid=False)
netD_A = networks.define_D(D_input_nc_A, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, use_sigmoid=False)

device = 'cpu'
if opt.cuda:
    netG_A.cuda()
    netG_B.cuda()
    netD_A.cuda()
    netD_B.cuda()
    if opt.use_geom==1:
        netGeom.cuda()
    device = 'cuda'

### load pretrained inception
net_recog = InceptionV3(opt.num_classes, opt.mode=='test', use_aux=True, pretrain=True, freeze=True, every_feat=opt.every_feat==1)
net_recog.cuda()
net_recog.eval()

import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
clip.model.convert_weights(clip_model)



#### load in progress weights if continue train or load_pretrain
if opt.continue_train:
    netG_A.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'netG_A_%s.pth' % opt.which_epoch)))
    netG_B.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'netG_B_%s.pth' % opt.which_epoch)))
    netD_A.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'netD_A_%s.pth' % opt.which_epoch)))
    netD_B.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'netD_B_%s.pth' % opt.which_epoch)))
    if opt.finetune_netGeom == 1:
        netGeom.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'netGeom_%s.pth'% opt.which_epoch)))
    print('----------- loaded %s from '%opt.which_epoch + os.path.join(checkpoints_dir, name) + '---------------------- !!')
elif len(opt.load_pretrain) > 0:
    pretrained_path = opt.load_pretrain
    netG_A.load_state_dict(torch.load(os.path.join(pretrained_path, 'netG_A_%s.pth' % opt.which_epoch)))
    netG_B.load_state_dict(torch.load(os.path.join(pretrained_path, 'netG_B_%s.pth' % opt.which_epoch)))
    netD_A.load_state_dict(torch.load(os.path.join(pretrained_path, 'netD_A_%s.pth' % opt.which_epoch)))
    netD_B.load_state_dict(torch.load(os.path.join(pretrained_path, 'netD_B_%s.pth' % opt.which_epoch)))
    if opt.finetune_netGeom == 1:
        netGeom.load_state_dict(torch.load(os.path.join(pretrained_path, 'netGeom_%s.pth'% opt.which_epoch)))
    print('----------- loaded %s from '%opt.which_epoch + ' ' + pretrained_path + '---------------------- !!')
else:
    netG_A.apply(weights_init_normal)
    netG_B.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)
    

print('----------- loaded networks ---------------------- !!')

# Losses

criterionGAN = networks.GANLoss(use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, reduceme=True).to(device)

criterion_MSE = torch.nn.MSELoss(reduce=True)
criterionCycle = torch.nn.L1Loss()
criterionCycleB = criterionCycle

criterionCLIP = criterion_MSE
if opt.cos_clip == 1:
    criterionCLIP = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

criterionGeom = torch.nn.BCELoss(reduce=True)

############### only use B to A ###########################
optimizer_G_A = torch.optim.Adam(netG_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_G_B = torch.optim.Adam(netG_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

if (opt.use_geom == 1 and opt.finetune_netGeom==1):
    optimizer_Geom = torch.optim.Adam(netGeom.parameters(), lr=opt.lr, betas=(0.5, 0.999))

optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G_A = torch.optim.lr_scheduler.LambdaLR(optimizer_G_A,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

lr_scheduler_G_B = torch.optim.lr_scheduler.LambdaLR(optimizer_G_B,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensorreal_A

# Dataset loader
transforms_r = [transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
               transforms.RandomCrop(opt.size),
               transforms.ToTensor()]

train_data = UnpairedDepthDataset(opt.dataroot, opt.root2, opt, transforms_r=transforms_r, 
                mode=opt.mode, midas=opt.midas>0, depthroot=opt.depthroot)

dataloader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu, drop_last=True)


print('---------------- loaded %d images ----------------' % len(train_data))


###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        total_steps = epoch*len(dataloader) + i

        img_r  = Variable(batch['r']).cuda()
        img_depth  = Variable(batch['depth']).cuda()

        real_A = img_r
        labels = Variable(batch['label']).cuda()

        real_B = 0
        real_B  = Variable(batch['line']).cuda()

        recover_geom = img_depth
        batch_size = real_A.size()[0]

        condGAN = opt.condGAN
        cond_recog = opt.cond_recog
        cond_cycle = opt.cond_cycle

        #################### Generator ####################


        fake_B = netG_A(real_A) # G_A(A)
        rec_A = netG_B(fake_B)   # G_B(G_A(A))

        fake_A = netG_B(real_B)  # G_B(B)
        rec_B = netG_A(fake_A) # G_A(G_B(B))

        loss_cycle_Geom = 0
        if opt.use_geom == 1:
            geom_input = fake_B
            if geom_input.size()[1] == 1:
                geom_input = geom_input.repeat(1, 3, 1, 1)
            _, geom_input = net_recog(geom_input)

            pred_geom = netGeom(geom_input)
            pred_geom = (pred_geom+1)/2.0 ###[-1, 1] ---> [0, 1]

            loss_cycle_Geom = criterionGeom(pred_geom, recover_geom)

        ########## loss A Reconstruction ##########

        loss_G_A = criterionGAN(netD_A(fake_A), True)

        # GAN loss D_B(G_B(B))
        loss_G_B = 0
        pred_fake_GAN = netD_B(fake_B)
        loss_G_B = criterionGAN(netD_B(fake_B), True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        loss_cycle_A = criterionCycle(rec_A, real_A)
        loss_cycle_B = criterionCycleB(rec_B, real_B)
        # combined loss and calculate gradients

        loss_GAN = loss_G_A + loss_G_B
        loss_RC = loss_cycle_A + loss_cycle_B

        loss_G = cond_cycle*loss_RC + condGAN*loss_GAN
        loss_G += opt.condGeom*loss_cycle_Geom


        ### semantic loss
        loss_recog = 0

        # renormalize mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        recog_real = real_A
        recog_real0 = (recog_real[:, 0, :, :].unsqueeze(1) - 0.48145466) / 0.26862954
        recog_real1 = (recog_real[:, 1, :, :].unsqueeze(1) - 0.4578275) / 0.26130258
        recog_real2 = (recog_real[:, 2, :, :].unsqueeze(1) - 0.40821073) / 0.27577711
        recog_real = torch.cat([recog_real0, recog_real1, recog_real2], dim=1)

        line_input = fake_B
        if opt.output_nc == 1:
            line_input_channel0 = (line_input - 0.48145466) / 0.26862954
            line_input_channel1 = (line_input - 0.4578275) / 0.26130258
            line_input_channel2 = (line_input - 0.40821073) / 0.27577711
            line_input = torch.cat([line_input_channel0, line_input_channel1, line_input_channel2], dim=1)

        patches_r = [torch.nn.functional.interpolate(recog_real, size=224)]  #The resize operation on tensor.
        patches_l = [torch.nn.functional.interpolate(line_input, size=224)]

        ## patch based clip loss
        if opt.N_patches > 1:
            patches_r2, patches_l2 = createNRandompatches(recog_real, line_input, opt.N_patches, opt.patch_size)
            patches_r += patches_r2
            patches_l += patches_l2

        loss_recog = 0
        for patchnum in range(len(patches_r)):

            real_patch = patches_r[patchnum]
            line_patch = patches_l[patchnum]

            feats_r = clip_model.encode_image(real_patch).detach()
            feats_line = clip_model.encode_image(line_patch)

            myloss_recog = criterionCLIP(feats_line, feats_r.detach())
            if opt.cos_clip == 1:
                myloss_recog = 1.0 - loss_recog
                myloss_recog = torch.mean(loss_recog)

            patch_factor = (1.0 / float(opt.N_patches))
            if patchnum == 0:
                patch_factor = 1.0
            loss_recog += patch_factor*myloss_recog
        
        loss_G += cond_recog* loss_recog


        optimizer_G_A.zero_grad()
        optimizer_G_B.zero_grad()
        if opt.finetune_netGeom == 1:
            optimizer_Geom.zero_grad()


        loss_G.backward()

        optimizer_G_A.step()
        optimizer_G_B.step()
        if opt.finetune_netGeom == 1:
            optimizer_Geom.step()

        ##########  Discriminator A ##########

        # Fake loss
        pred_fake_A = netD_A(fake_A.detach())
        loss_D_A_fake = criterionGAN(pred_fake_A, False)

        # Real loss

        pred_real_A = netD_A(real_A)
        loss_D_A_real = criterionGAN(pred_real_A, True)

        # Total loss
        loss_D_A = torch.mean(condGAN * (loss_D_A_real + loss_D_A_fake) ) * 0.5

        optimizer_D_A.zero_grad()
        loss_D_A.backward()
        optimizer_D_A.step()

        ##########  Discriminator B ##########

        # Fake loss
        pred_fake_B = netD_B(fake_B.detach())
        loss_D_B_fake = criterionGAN(pred_fake_B, False)

        # Real loss

        pred_real_B = netD_B(real_B)
        loss_D_B_real = criterionGAN(pred_real_B, True)

        # Total loss
        loss_D_B = torch.mean(condGAN * (loss_D_B_real + loss_D_B_fake) ) * 0.5
        optimizer_D_B.zero_grad()
        loss_D_B.backward()
        optimizer_D_B.step()

        # Progress report
        if (i+1)%opt.log_int==0:

            errors = {}
            errors['total_G'] = loss_G.item() if not isinstance(loss_G, (int,float)) else loss_G
            errors['loss_RC'] = torch.mean(loss_RC) if not isinstance(loss_RC, (int,float)) else loss_RC
            errors['loss_cycle_Geom'] = torch.mean(loss_cycle_Geom) if not isinstance(loss_cycle_Geom, (int,float)) else loss_cycle_Geom
            errors['loss_GAN'] = torch.mean(loss_GAN) if not isinstance(loss_GAN, (int,float)) else loss_GANB
            errors['loss_D_B'] = loss_D_B.item() if not isinstance(loss_D_B, (int,float)) else loss_D_B
            errors['loss_D_A'] = loss_D_A.item() if not isinstance(loss_D_A, (int,float)) else loss_D_A
            errors['loss_recog'] = torch.mean(loss_recog) if not isinstance(loss_recog, (int,float)) else loss_recog
           
            visualizer.print_current_errors(epoch, total_steps, errors, 0)
            visualizer.plot_current_errors(errors, total_steps)

            with torch.no_grad():

                input_img = channel2width(real_A)
                if opt.use_geom == 1:
                    pred_geom = channel2width(pred_geom)
                    input_img = torch.cat([input_img, channel2width(recover_geom)], dim=3)
                    
                input_img_fake = channel2width(fake_A)
                rec_A = channel2width(rec_A)

                show_real_B = real_B

                visuals = OrderedDict([('real_A', tensor2im(input_img.data[0])),
                                           ('real_B', tensor2im(show_real_B.data[0])),
                                           ('fake_A', tensor2im(input_img_fake.data[0])),
                                           ('rec_A', tensor2im(rec_A.data[0])),
                                           ('fake_B', tensor2im(fake_B.data[0]))])

                if opt.use_geom == 1:
                    visuals['pred_geom'] = tensor2im(pred_geom.data[0])

                visualizer.display_current_results(visuals, total_steps, epoch)


    # Update learning rates
    lr_scheduler_G_A.step()
    lr_scheduler_G_B.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    # torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
    if (epoch+1) % opt.save_epoch_freq == 0:
        torch.save(netG_A.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netG_A_%02d.pth'%(epoch)))
        if opt.finetune_netGeom == 1:
            torch.save(netGeom.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netGeom_%02d.pth'%(epoch)))
        if opt.slow == 0:
            torch.save(netG_B.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netG_B_%02d.pth'%(epoch)))
            torch.save(netD_A.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netD_A_%02d.pth'%(epoch)))
            torch.save(netD_B.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netD_B_%02d.pth'%(epoch)))

    torch.save(netG_A.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netG_A_latest.pth'))
    torch.save(netG_B.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netG_B_latest.pth'))
    torch.save(netD_B.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netD_B_latest.pth'))
    torch.save(netD_A.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netD_A_latest.pth'))
    if opt.finetune_netGeom == 1:
        torch.save(netGeom.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netGeom_latest.pth'))

###################################
