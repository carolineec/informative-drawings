#!/usr/bin/python3

import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from model import Generator, GlobalGenerator2
from net_canny import CannyNet
# from utils import ReplayBuffer
from utils import LambdaLR
# from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset
# from aligned_dataset import AlignedDataset, LineDrawings
from dataset_caroline import LineDrawings, LineDrawings_sketch, NeuralContours, LineDrawingsPlusPlus, ImageDataset_styles
import utils_pl
from collections import OrderedDict
import util.util as util
# from util.visualizer import Visualizer
from vgg import Vgg16
import networks
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help='name of this experiment')
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')

parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/vangogh2photo/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=10,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--geom_nc', type=int, default=6, help='number of channels of output data')
parser.add_argument('--line_nc', type=int, default=1, help='number of channels of output data')

parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
parser.add_argument('--netD', type=str, default='basic', help='selects model to use for netD')
parser.add_argument('--netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')

parser.add_argument('--load_pretrain', type=str, default='', help='where to load file if wanted')
parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load from if continue_train')

parser.add_argument('--mode', type=str, default='train', help='train, val, test, etc')
parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

parser.add_argument('--conditional_GAN', action='store_true', help='use conditional GAN', default=True)

######## CONDS
parser.add_argument('--cond_size', type=int, default=2, help='sym parameter size')
parser.add_argument('--no_gates', type=int, default=0, help='do not use sym parameter')
parser.add_argument('--YOTO', type=int, default=0, help='1 = use YOTO gates, 2 = YOTO everywhere')
parser.add_argument('--spacial', type=int, default=0, help='use spacial sym parameter map')
parser.add_argument('--canny_out', type=int, default=0, help='dont use this')
parser.add_argument("--cond0", type=float, default=1.0)
parser.add_argument("--cond1", type=float, default=1.0)
parser.add_argument("--cond2", type=float, default=1.0)
parser.add_argument("--cond3", type=float, default=1.0)
parser.add_argument("--cond_fp", type=float, default=1.0)
parser.add_argument("--cycle_mult", type=float, default=1.0)
parser.add_argument("--stages", type=int, default=1)
parser.add_argument("--n_stages", type=int, default=30)
parser.add_argument("--cycle_cheat", type=int, default=30)
parser.add_argument('--usegan', type=int, default=0, help='use gan loss')
parser.add_argument("--cond_gan", type=float, default=1.0)
parser.add_argument('--canny_net', type=int, default=0, help='use canny net')
parser.add_argument('--thin', type=int, default=0, help='thin edges')
parser.add_argument('--thresh', type=int, default=0, help='use YOTO gates')
parser.add_argument('--fixed_point', type=int, default=0, help='use fixed point loss')
parser.add_argument('--mask', type=int, default=0, help='use masked sparsity loss')
parser.add_argument('--round', type=int, default=0, help='use rounding (experimental)')
parser.add_argument('--middle', type=float, default=0.0, help='losses on middle man')
parser.add_argument('--upsample', type=int, default=0, help='upsample instead of convT')
parser.add_argument('--texture', type=int, default=0, help='use texture instead of hed')
parser.add_argument('--cond_text', type=float, default=1.0, help='weight on texture')
parser.add_argument('--num_curves', type=int, default=0, help='number of curves to take as input')
parser.add_argument('--radius', type=int, default=-1, help='curve radius')
parser.add_argument('--lines_only', type=int, default=0, help='lines only')
parser.add_argument('--rf_scale', type=float, default=1.0, help='resize the image for different discrim receptive field')
parser.add_argument('--just_geom', type=int, default=0, help='use depth and nv maps')
parser.add_argument('--cheating', type=int, default=0, help='cheating with texture GAN')
parser.add_argument('--recover', type=int, default=0, help='what to make cycle with')
parser.add_argument('--thicc', type=int, default=2, help='line thickness')
parser.add_argument('--finetune', type=int, default=0, help='finetune last layer of inceptioon')
parser.add_argument('--cond_GAN', type=float, default=0.0)
parser.add_argument('--uselines', type=int, default=0, help='liine drawing as input')
parser.add_argument('--use_canny', type=int, default=0, help='canny line drawing as input')
parser.add_argument('--mesh', type=int, default=0, help='use mesh')
parser.add_argument('--objects', type=int, default=0, help='idk')
parser.add_argument('--midas', type=int, default=0, help='use just midas depth map')
parser.add_argument('--multistyle', type=int, default=0, help='use multiple style examples')

parser.add_argument('--norm_G', type=str, default='spectralspadesyncbatch3x3', help='instance normalization or batch normalization')

parser.add_argument('--corners', type=int, default=0, help='percentage of time to sample corners')
parser.add_argument('--edges', type=int, default=0, help='sample corners 1 = half time, 2 = all the time')

parser.add_argument('--load_pairs', type=int, default=0, help='load pairs even if unaligned data')

parser.add_argument('--sparsity', type=int, default=0, help='add a sparsity loss')
parser.add_argument('--recognize', type=int, default=0, help='add a recognizability loss')
parser.add_argument('--hed', type=int, default=0, help='add hed network to end of thing')
parser.add_argument('--optimize_hed', type=int, default=0, help='optimize hed network')
parser.add_argument('--hed_lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='initial learning rate')


parser.add_argument('--name_recog', type=str, default='shapenet_r_classifier_inception', help='name of the recognizability experiment')
parser.add_argument('--dataset', type=str, default='aligned', help='aligned or unaligned')
parser.add_argument('--root2', type=str, default='', help='for unaligned datasets')

parser.add_argument('--num_classes', type=int, default=55, help='number of classes for inception')


parser.add_argument('--cuda', action='store_true', help='use GPU computation', default=True)
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--log_int', type=int, default=50, help='number of cpu threads to use during batch generation')
parser.add_argument('--style_image', type=str, default='images/style-images/udnie.jpg', help='root directory of the dataset')
parser.add_argument("--rc_weight", type=float, default=2.,
                                  help="reconstruction weight, default is 2")
parser.add_argument("--identity_weight", type=float, default=5.,
                                  help="weight for identity-loss, default is 5.")

parser.add_argument("--content_weight", type=float, default=1e-3,
                                  help="weight for content-loss, default is 1e-3")
parser.add_argument("--style_weight", type=float, default=1e2,
                                  help="weight for style-loss, default is 1e2")
parser.add_argument("--style_size", type=int, default=256,
                                  help="size of style-image, default is the original size of style image")

parser.add_argument("--alpha", type=float, default=0.5,
                                  help="alpha for dirichlet")
parser.add_argument('--checkpoints_dir', type=str, default='/afs/csail.mit.edu/u/c/cmchan/experiments', \
                help='Where checkpoints are saved')
parser.add_argument('--data_order', nargs='+', default=['B', 'A'])
parser.add_argument('--save_epoch_freq', type=int, default=100, help='how often to save the latest model in steps')

opt = parser.parse_args()
print(opt)

checkpoints_dir = opt.checkpoints_dir #'/afs/csail.mit.edu/u/c/cmchan/experiments'
name = opt.name #'testing2d'

tensor2im = util.tensor2im
if opt.midas > 0:
    from util.visualizer2 import Visualizer
    tensor2im = util.tensor2imv2
else:
    from util.visualizer import Visualizer

visualizer = Visualizer(checkpoints_dir, name, tf_log=True, isTrain=True)
print('------------------- created visualizer -------------------')

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks

netG_A = 0
mm = opt.input_nc
if opt.uselines:
  mm = opt.line_nc
if opt.recognize==1:
  mm = 768
if opt.recognize==2:
  mm = 512

# if opt.no_gates == 1:
#     netG_A = Generator_noCCAM_sigmoid(mm, opt.geom_nc, opt.cond_size)
#     # netG_B = Generator_noCCAM_sigmoid(opt.line_nc, opt.geom_nc, opt.cond_size)
#     if opt.upsample == 1:
#         netG_A = Generator_noCCAM_sigmoid_upsample(opt.input_nc, opt.output_nc, opt.cond_size)
#     if opt.recognize == 2:
#         netG_A = GlobalGeneratorfromz(mm, opt.ngf, opt.output_nc, h_size=mm)
# elif opt.no_gates == 2:
#     netG_A = GlobalGenerator2(mm, opt.geom_nc, n_downsampling=1, n_UPampling=3)
# elif opt.no_gates == 3:
#     netG_A = GlobalGeneratorfromz(mm, opt.geom_nc, n_downsampling=0, n_UPampling=7, use_sig=False)
    # netG_A = StyleGAN2Generator(mm, opt.output_nc, opt.ngf, use_dropout=True, num_UPsamplingSv2=5, insize=16,outsize=256)
netG_A = GlobalGenerator2(mm, opt.geom_nc, n_downsampling=1, n_UPampling=3)

D_input_nc = opt.input_nc + opt.geom_nc
netD = NLayerDiscriminator_Gates_Scale(D_input_nc, opt.ndf, opt.n_layers_D, norm=opt.norm, use_sigmoid=False, scale=opt.rf_scale)


if opt.recognize==1: #### use imagenet feats
    numclasses=69
    net_recog = Inception_bare(numclasses, opt.mode=='test', use_aux=True, use_gates=False, finetune=True, freeze=True, feats=True, every_feat=True)
    net_recog.cuda()
    net_recog.eval()
if opt.recognize==2: ### CLIP LOSS
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.float()
        # netG_B = Generator_noCCAM_sigmoid_upsample(opt.line_nc, opt.geom_nc, opt.cond_size)
# elif opt.YOTO == 1:
#     netG_A = GeneratorYOTO_sigmoid(opt.input_nc, opt.output_nc, opt.cond_size)
#     netG_B = Generator_noCCAM_sigmoid(opt.output_nc, opt.input_nc, opt.cond_size)
# elif opt.YOTO == 2:
#     netG_A = GeneratorYOTO_everywhere_sigmoid(opt.input_nc, opt.output_nc, opt.cond_size)
#     netG_B = Generator_noCCAM_sigmoid(opt.output_nc, opt.input_nc, opt.cond_size)
# elif opt.YOTO == 3:
#     netG_A = SPADEGenerator(opt.input_nc, opt.output_nc, opt.cond_size, opt, seg_input=opt.spacial==1)
#     netG_B = Generator_noCCAM_sigmoid(opt.output_nc, opt.input_nc, opt.cond_size)
# else:
#     netG_A = Generator_sigmoid(opt.input_nc, opt.output_nc, opt.cond_size)
#     netG_B = Generator_sigmoid(opt.output_nc, opt.input_nc, opt.cond_size)

# netG_A = Generator_sigmoid(opt.input_nc, opt.output_nc, opt.cond_size)
# netG_B = Generator_sigmoid(opt.output_nc, opt.input_nc, opt.cond_size)
# netG = GeneratorNORM(opt.output_nc, opt.input_nc)
# netG = GeneratorCat(opt.output_nc, opt.input_nc)


#### load in progress weights if continue train or load_pretrain
if opt.continue_train:
    netG_A.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'netG_A_%s.pth' % opt.which_epoch)))
    netD.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'netD_%s.pth' % opt.which_epoch)))
    print('----------- loaded %s from '%opt.which_epoch + os.path.join(checkpoints_dir, name) + '---------------------- !!')
elif len(opt.load_pretrain) > 0:
    pretrained_path = opt.load_pretrain
    netG_A.load_state_dict(torch.load(os.path.join(pretrained_path, 'netG_A_%s.pth' % opt.which_epoch)))
    netD.load_state_dict(torch.load(os.path.join(pretrained_path, 'netD_%s.pth' % opt.which_epoch)))

    print('----------- loaded %s from '%opt.which_epoch + ' ' + pretrained_path + '---------------------- !!')

print('----------- loaded networks ---------------------- !!')


device = 'cpu'
if opt.cuda:
    netG_A.cuda()
    netD.cuda()
    device = 'cuda'

if opt.no_gates < 2:
    netG_A.apply(weights_init_normal)

# Losses

criterionGAN = networks.GANLoss().to(device)
criterion_MSE = torch.nn.MSELoss(reduce=False)
criterionCycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
criterion_BCE = torch.nn.BCELoss(reduce=False)
criterionCrossEntropy = torch.nn.CrossEntropyLoss(reduce=False)
# vggloss = VGGLoss()


############### only use B to A ###########################
optimizer_G_A = torch.optim.Adam(netG_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
lr_scheduler_G_A = torch.optim.lr_scheduler.LambdaLR(optimizer_G_A,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

optimizer_D = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
# optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D,
                                                 lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensorreal_A

# fake_A_buffer = ReplayBuffer()
# fake_B_buffer = ReplayBuffer()
# fake_C_buffer = ReplayBuffer()
# Dataset loader
transforms_r = [transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
               transforms.RandomCrop(opt.size),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor()]
               # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transforms_d = [transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
               transforms.RandomCrop(opt.size),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor()]

if opt.dataset == 'aligned':
    train_data = LineDrawings(opt.dataroot, opt, transforms_r=transforms_r, mode=opt.mode, curves=opt.num_curves, radius=opt.radius, lines=opt.lines_only, root2=opt.root2, mesh=opt.mesh==1)
    if opt.use_canny==1:
      train_data = LineDrawingsPlusPlus(opt.dataroot, opt, transforms_r=transforms_r, mode=opt.mode)
elif opt.dataset == 'neuralcontours':
    train_data = NeuralContours(opt.dataroot, opt, transforms_r=transforms_r, mode=opt.mode)
elif opt.dataset == 'images':
    multistyle = opt.multistyle
    train_data = ImageDataset_styles(opt.dataroot, opt.root2, opt, transforms_r=transforms_r, mode=opt.mode, islist=opt.root2=='artists', \
                        load_pairs=opt.load_pairs==1, multistyle=multistyle, use_captions=True, midas=opt.midas>0)
    num_styles = train_data.get_numstyles()
    # if num_styles != opt.num_styles:
    #     print('please set --num_styles to the correct amount %d' %num_styles)
    #     import sys
    #     sys.exit(0)
else:
    train_data = LineDrawings_sketch(opt.dataroot, opt.root2, opt, transforms_r=transforms_r, mode=opt.mode, islist=opt.root2=='artists', load_pairs=opt.load_pairs==1, curves=opt.num_curves)

dataloader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu, drop_last=True)


print('---------------- loaded %d images ----------------' % len(train_data))
###################################


###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        total_steps = epoch*len(dataloader) + i

        real_B  = Variable(batch['line']).cuda()

        img_r  = Variable(batch['r']).cuda()
        img_depth  = Variable(batch['depth']).cuda()
        img_normals  = Variable(batch['normals']).cuda()
        mask  = img_r #Variable(batch['mask']).cuda()
        # img_curves  = Variable(batch['curves']).cuda()

        img_mesh = 0 #Variable(batch['mesh']).cuda()
        
        real_A = 0
        labels = Variable(batch['label']).cuda()
        real_A = img_r

        if opt.just_geom == 1:
            real_A = torch.cat([img_depth, img_normals], dim=1)

        recover_geom = img_depth
        # recover_geom = torch.cat([img_depth, img_normals], dim=1)
        # if opt.recover == 1:
        #     recover_geom = img_r
        # elif opt.recover == 2:
        #     recover_geom = torch.cat([img_r, img_depth, img_normals], dim=1)
        # elif opt.recover == 3:
        #     recover_geom = img_depth
        # if opt.mesh == 1:
        #     recover_geom = torch.cat([recover_geom, img_mesh], dim=1)

        # print(torch.max(real_A), torch.min(real_A), torch.max(real_B), torch.min(real_B))

        batch_size = real_A.size()[0]

        cond = 0
        #################### Generator ####################

        uhhh = real_A #[:, :3, :, :]
        justforkicks = real_B

        inputme = real_A

        if opt.recognize ==1: ### FEATURE LOSS
            ### renormalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            recog_real = real_A
            _, feats_r = net_recog(recog_real)
            inputme = feats_r
        elif opt.recognize ==2: ### CLIP FEATURE LOSS
            ### renormalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            recog_real = torch.nn.functional.interpolate(real_A, size=224)
            feats_clip = clip_model.encode_image(recog_real)
            # print(feats_clip)
            inputme = feats_clip
        if opt.uselines == 1:
            if opt.use_canny == 1:
                real_B  = Variable(batch['canny']).cuda()
            inputme = real_B
        fake_A = netG_A(inputme) # G_A(A)

        # print(inputme.size(), 'insize')
        # print(fake_A.size(), 'outsize')
        # print(recover_geom.size(), 'mysize')
        loss_cycle = torch.mean(criterionCycle(fake_A, recover_geom)) #* lambda_A

        loss_GAN = 0
        if opt.usegan == 1:
            # loss_G_A = criterionGAN(netD_A(fake_A), True)
            # GAN loss D_B(G_B(B))
            fake_input = torch.cat([real_A, fake_A], dim=1)
            loss_GAN = criterionGAN(netD(fake_input), True)
            # Forward cycle loss || G_B(G_A(A)) - A||
            loss_GAN = torch.mean(loss_GAN)

        # vgg = VGGLoss(fake_A, recover_geom)

        loss_G = loss_cycle + opt.cond_GAN*loss_GAN

        optimizer_G_A.zero_grad()
        loss_G.backward()
        optimizer_G_A.step()

        loss_D = 0
        if opt.usegan == 1:

            # Fake loss
            fake_input_D = torch.cat([real_A, fake_A], dim=1)
            pred_fake_B = netD(fake_input_D.detach())
            loss_D_B_fake = criterionGAN(pred_fake_B, False)

            # Real loss

            real_input_D = torch.cat([real_A, recover_geom], dim=1)
            pred_real_B = netD(real_input_D)
            loss_D_B_real = criterionGAN(pred_real_B, True)


            # Total loss
            loss_D = torch.mean((loss_D_B_real + loss_D_B_fake) ) * 0.5
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
    
        # print(torch.min(real_B), torch.max(real_B), 'real')
        # print(torch.min(out_im), torch.max(out_im), 'out')

        # Progress report (http://localhost:8097)
        if (i+1)%opt.log_int==0:

            # print(fake_B_before, 'before')
            # print(fake_B, 'after')


            errors = {}
            errors['total_G'] = loss_G.item() if not isinstance(loss_G, (int,float)) else loss_G
            errors['loss_GAN'] = loss_GAN.item() if not isinstance(loss_GAN, (int,float)) else loss_GAN
            errors['loss_D'] = loss_D.item() if not isinstance(loss_D, (int,float)) else loss_D
            errors['loss_RC'] = loss_cycle.item() if not isinstance(loss_cycle, (int,float)) else loss_cycle
            # errors['loss_sparsity'] = torch.mean(loss_sparsity) if not isinstance(loss_sparsity, (int,float)) else loss_sparsity
            # errors['loss_recog'] = torch.mean(loss_recog) if not isinstance(loss_recog, (int,float)) else loss_recog
            # errors['loss_fp'] = torch.mean(loss_fp) if not isinstance(loss_fp, (int,float)) else loss_fp
            # errors['loss_middle'] = torch.mean(loss_middle) if not isinstance(loss_middle, (int,float)) else loss_middle
            # errors['loss_texture'] = torch.mean(loss_texture) if not isinstance(loss_texture, (int,float)) else loss_texture

            visualizer.print_current_errors(epoch, total_steps, errors, 0)
            visualizer.plot_current_errors(errors, total_steps)

            with torch.no_grad():
                # input_img = torch.cat([img_depth, img_normals], dim=3)
                # if (opt.dataset != 'neuralcontours') or (opt.just_geom == 0):
                #     input_img = torch.cat([img_r, input_img], dim=3)
                # if opt.recover == 1:
                #     input_img_fake = fake_A
                #     # rec_A_view = rec_A
                # elif opt.recover == 2:
                #     input_img_fake = torch.cat([fake_A[:, :3, :, :], fake_A[:, 3:6, :, :], fake_A[:, 6:, :, :]], dim=3)
                #     # rec_A_view = torch.cat([rec_A[:, :3, :, :], rec_A[:, 3:6, :, :], rec_A[:, 6:, :, :]], dim=3)
                # else:
                #     img_depth_fake = fake_A[:, :3, :, :]
                #     img_normals_fake = fake_A[:, 3:6, :, :]
                #     input_img_fake = torch.cat([img_depth_fake, img_normals_fake], dim=3)
                #     if (opt.dataset != 'neuralcontours') or (opt.just_geom == 0):
                #         img_depth_fake = fake_A[:, :3, :, :]
                #         img_normals_fake = fake_A[:, 3:6, :, :]
                #         # img_normals_fake = fake_A[:, 6:, :, :]
                #         input_img_fake = torch.cat([img_depth_fake, img_normals_fake], dim=3)
                #     # rec_A_view = torch.cat([rec_A[:, :3, :, :], rec_A[:, 3:, :, :]], dim=3)
                #     if opt.mesh == 1:
                #         input_img_fake = torch.cat([input_img_fake, fake_A[:, 6:, :, :]], dim=3)

                input_img_fake = fake_A
                input_img = img_r

                if opt.mesh == 1:
                    real_B = torch.cat([real_B, img_mesh], dim=3)

                all_together = torch.cat([input_img, fake_A, recover_geom], dim=3)

                visuals = OrderedDict([('real_A', tensor2im(input_img.data[0])),
                                           ('real_B', tensor2im(recover_geom.data[0])),
                                           ('all_together', tensor2im(all_together.data[0])),
                                           ('mask', tensor2im(mask.data[0])),
                                           ('fake_A', tensor2im(input_img_fake.data[0]))])

                visualizer.display_current_results(visuals, total_steps, epoch)


    # Update learning rates
    lr_scheduler_G_A.step()


    # Save models checkpoints
    # torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
    if (epoch+1) % opt.save_epoch_freq == 0:
        torch.save(netG_A.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netG_A_%02d.pth'%(epoch)))
        torch.save(netD.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netD_%02d.pth'%(epoch)))

    torch.save(netG_A.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netG_A_latest.pth'))
    torch.save(netD.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netD_latest.pth'))
###################################
