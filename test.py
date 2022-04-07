#!/usr/bin/python3

import argparse
import os
import sys

import coremltools as ct
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import UnpairedDepthDataset, VideoDataset
from model import Generator, GlobalGenerator2, InceptionV3
from utils import channel2width


CT_INPUT_NAME = "input"


def create_generators(pt_checkpoint_path, device, size, use_coreml=False, **kwargs):
    """Creates models: pytorch and, optionally, corresponding coreml"""
    if use_coreml:
        coreml_checkpoint_path = f"{pt_checkpoint_path:s}.mlmodel"
        if os.path.exists(coreml_checkpoint_path):
            ct_model = ct.models.MLModel(coreml_checkpoint_path)
        else:
            pt_model = create_pt_generator(pt_checkpoint_path, device, **kwargs)
            ct_model = convert_to_coreml(pt_model, size)
            ct_model.save(coreml_checkpoint_path)
        return None, ct_model
    pt_model = create_pt_generator(pt_checkpoint_path, device, **kwargs)
    return pt_model, None


def create_pt_generator(checkpoint_path, device, **kwargs):
    """Creates the generator model in pytorch"""
    pt_model = Generator(**kwargs).eval()
    pt_model.to(device)
    print(
        pt_model.load_state_dict(
            torch.load(checkpoint_path, map_location=device), strict=False
        )
    )
    return pt_model


def convert_to_coreml(model, size):
    """Converts pytorch model to coreml"""
    inputs = torch.randn(1, 3, size, size)
    traced_model = torch.jit.trace(model, inputs)
    ct_input = ct.TensorType(
        name=CT_INPUT_NAME,
        shape=ct.Shape(
            shape=(
                ct.RangeDim(),
                3,
                ct.RangeDim(),
                ct.RangeDim(),
            ),
        ),
    )
    return ct.convert(
        traced_model,
        inputs=[ct_input],
    )


def run_generator(pt_model, coreml_model, input_tensor):
    """Runs generator model. Runs coreml if provided, otherwise pytorch"""
    if coreml_model is None:
        image_tensor = pt_model(input_tensor)
    else:
        # Since there is a single output, we can just subset ".values";
        # in the multi-output case, extracting the output by its name is recommended.
        image = list(
            coreml_model.predict({CT_INPUT_NAME: input_tensor.cpu().numpy()}).values()
        )[-1]
        image_tensor = torch.from_numpy(image)
    return image_tensor


parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, type=str, help="name of this experiment")
parser.add_argument(
    "--checkpoints_dir",
    type=str,
    default="checkpoints",
    help="Where the model checkpoints are saved",
)
parser.add_argument(
    "--results_dir", type=str, default="results", help="where to save result images"
)
parser.add_argument(
    "--geom_name", type=str, default="feats2Geom", help="name of the geometry predictor"
)
parser.add_argument("--batchSize", type=int, default=1, help="size of the batches")
parser.add_argument(
    "--dataroot", type=str, default="", help="root directory of the dataset"
)
parser.add_argument(
    "--depthroot",
    type=str,
    default="",
    help="dataset of corresponding ground truth depth maps",
)

parser.add_argument(
    "--input_nc", type=int, default=3, help="number of channels of input data"
)
parser.add_argument(
    "--output_nc", type=int, default=1, help="number of channels of output data"
)
parser.add_argument(
    "--geom_nc", type=int, default=3, help="number of channels of geometry data"
)
parser.add_argument(
    "--every_feat",
    type=int,
    default=1,
    help="use transfer features for the geometry loss",
)
parser.add_argument(
    "--num_classes", type=int, default=55, help="number of classes for inception"
)
parser.add_argument("--midas", type=int, default=0, help="use midas depth map")

parser.add_argument(
    "--ngf", type=int, default=64, help="# of gen filters in first conv layer"
)
parser.add_argument(
    "--n_blocks", type=int, default=3, help="number of resnet blocks for generator"
)
parser.add_argument(
    "--size", type=int, default=256, help="size of the data (squared assumed)"
)
parser.add_argument(
    "--cuda", action="store_true", help="use GPU computation", default=True
)
parser.add_argument(
    "--n_cpu",
    type=int,
    default=8,
    help="number of cpu threads to use during batch generation",
)
parser.add_argument(
    "--which_epoch", type=str, default="latest", help="which epoch to load from"
)
parser.add_argument(
    "--aspect_ratio",
    type=float,
    default=1.0,
    help="The ratio width/height. The final height of the load image will be crop_size/aspect_ratio",
)

parser.add_argument("--mode", type=str, default="test", help="train, val, test, etc")
parser.add_argument(
    "--load_size", type=int, default=256, help="scale images to this size"
)
parser.add_argument("--crop_size", type=int, default=256, help="then crop to this size")
parser.add_argument(
    "--max_dataset_size",
    type=int,
    default=float("inf"),
    help="Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.",
)
parser.add_argument(
    "--preprocess",
    type=str,
    default="resize_and_crop",
    help="scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]",
)
parser.add_argument(
    "--no_flip",
    action="store_true",
    help="if specified, do not flip the images for data augmentation",
)
parser.add_argument(
    "--norm",
    type=str,
    default="instance",
    help="instance normalization or batch normalization",
)

parser.add_argument(
    "--predict_depth",
    type=int,
    default=0,
    help="run geometry prediction on the generated images",
)
parser.add_argument("--save_input", type=int, default=0, help="save input image")
parser.add_argument("--reconstruct", type=int, default=0, help="get reconstruction")
parser.add_argument(
    "--how_many", type=int, default=100, help="number of images to test"
)
parser.add_argument("--video", action="store_true", help="use video dataset")
parser.add_argument("--coreml", action="store_true", help="use coreml for inference")

opt = parser.parse_args()
print(opt)

opt.no_flip = True

# By default, will run on CPU unless CUDA is requested
device = torch.device("cpu")
is_cuda_available = torch.cuda.is_available()
if opt.cuda:
    if is_cuda_available:
        device = torch.device("cuda")
    else:
        print(
            "WARNING: You requested a CUDA device, but CUDA is not available -- using CPU instead"
        )
elif is_cuda_available:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

with torch.no_grad():
    # Networks

    net_G, net_G_coreml = create_generators(
        os.path.join(opt.checkpoints_dir, opt.name, "netG_A_%s.pth" % opt.which_epoch),
        device,
        opt.size,
        use_coreml=opt.coreml,
        input_nc=opt.input_nc,
        output_nc=opt.output_nc,
        n_residual_blocks=opt.n_blocks,
    )

    net_GB = 0
    if opt.reconstruct == 1:
        if opt.coreml:
            raise ValueError("Cannot yet run reconstruction with coreml")
        net_GB = Generator(opt.output_nc, opt.input_nc, opt.n_blocks)
        net_GB.to(device)
        net_GB.load_state_dict(
            torch.load(
                os.path.join(
                    opt.checkpoints_dir, opt.name, "netG_B_%s.pth" % opt.which_epoch
                ),
                map_location=device,
            )
        )
        net_GB.eval()

    netGeom = 0
    if opt.predict_depth == 1:
        if opt.coreml:
            raise ValueError("Cannot yet run depth prediction with coreml")
        usename = opt.name
        if (len(opt.geom_name) > 0) and (
            os.path.exists(os.path.join(opt.checkpoints_dir, opt.geom_name))
        ):
            usename = opt.geom_name
        myname = os.path.join(
            opt.checkpoints_dir, usename, "netGeom_%s.pth" % opt.which_epoch
        )
        netGeom = GlobalGenerator2(768, opt.geom_nc, n_downsampling=1, n_UPsampling=3)

        netGeom.load_state_dict(torch.load(myname, map_location=device))
        netGeom.to(device)
        netGeom.eval()

        numclasses = opt.num_classes
        ### load pretrained inception
        net_recog = InceptionV3(
            opt.num_classes,
            False,
            use_aux=True,
            pretrain=True,
            freeze=True,
            every_feat=opt.every_feat == 1,
        )
        net_recog.to(device)
        net_recog.eval()

    transforms_r = [
        transforms.Resize((int(opt.size), int(opt.size)), Image.BICUBIC),
        transforms.ToTensor(),
    ]

    dataset_cls = UnpairedDepthDataset
    if opt.video:
        dataset_cls = VideoDataset
    test_data = dataset_cls(
        opt.dataroot,
        "",
        opt,
        transforms_r=transforms_r,
        mode=opt.mode,
        midas=opt.midas > 0,
        depthroot=opt.depthroot,
    )

    dataloader = DataLoader(test_data, batch_size=opt.batchSize, shuffle=False)

    ###################################

    ###### Testing######

    full_output_dir = os.path.join(opt.results_dir, opt.name)

    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)

    for i, batch in enumerate(dataloader):
        if i > opt.how_many:
            break
        img_r = Variable(batch["r"]).to(device)
        img_depth = Variable(batch["depth"]).to(device)
        real_A = img_r

        name = batch["name"][0]

        input_image = real_A
        image = run_generator(net_G, net_G_coreml, input_image)
        save_image(image.data, full_output_dir + "/%s_out.png" % name)

        if opt.predict_depth == 1:

            geom_input = image
            if geom_input.size()[1] == 1:
                geom_input = geom_input.repeat(1, 3, 1, 1)
            _, geom_input = net_recog(geom_input)
            geom = netGeom(geom_input)
            geom = (geom + 1) / 2.0  ###[-1, 1] ---> [0, 1]

            input_img_fake = channel2width(geom)
            save_image(input_img_fake.data, full_output_dir + "/%s_geom.png" % name)

        if opt.reconstruct == 1:
            rec = net_GB(image)
            save_image(rec.data, full_output_dir + "/%s_rec.png" % name)

        if opt.save_input == 1:
            save_image(img_r, full_output_dir + "/%s_input.png" % name)

        sys.stdout.write("\rGenerated images %04d of %04d" % (i, opt.how_many))

    sys.stdout.write("\n")
    ###################################
