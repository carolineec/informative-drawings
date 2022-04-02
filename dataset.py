import bisect
import os

import cv2
import decord
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data

from base_dataset import get_params, get_transform

IMG_EXTENSIONS = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]
VIDEO_EXTENSIONS = [
    ".mp4",
]


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VIDEO_EXTENSIONS)


def make_dataset(dir, stop=10000, filter_fn=is_image_file):
    images = []
    count = 0
    assert os.path.isdir(dir), "%s is not a valid directory" % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if filter_fn(fname):
                path = os.path.join(root, fname)
                images.append(path)
                count += 1
            if count >= stop:
                return images
    return images


class UnpairedDepthDataset(data.Dataset):
    def __init__(
        self,
        root,
        root2,
        opt,
        transforms_r=None,
        mode="train",
        midas=False,
        depthroot="",
    ):

        self.root = root
        self.mode = mode
        self.midas = midas

        all_img = make_dataset(self.root)

        self.depth_maps = 0
        if self.midas:

            depth = []
            print(depthroot)
            if os.path.exists(depthroot):
                depth = make_dataset(depthroot)
            else:
                print("could not find %s" % depthroot)
                import sys

                sys.exit(0)

            newimages = []
            self.depth_maps = []

            for dmap in depth:
                lastname = os.path.basename(dmap)
                trainName1 = os.path.join(self.root, lastname)
                trainName2 = os.path.join(self.root, lastname.split(".")[0] + ".jpg")
                if os.path.exists(trainName1):
                    newimages += [trainName1]
                elif os.path.exists(trainName2):
                    newimages += [trainName2]
            print("found %d correspondences" % len(newimages))

            self.depth_maps = depth
            all_img = newimages

        self.data = all_img
        self.mode = mode

        self.transform_r = transforms.Compose(transforms_r)

        self.opt = opt

        if mode == "train":

            self.img2 = make_dataset(root2)

            if len(self.data) > len(self.img2):
                howmanyrepeat = (len(self.data) // len(self.img2)) + 1
                self.img2 = self.img2 * howmanyrepeat
            elif len(self.img2) > len(self.data):
                howmanyrepeat = (len(self.img2) // len(self.data)) + 1
                self.data = self.data * howmanyrepeat
                self.labels = self.labels * howmanyrepeat

            cutoff = min(len(self.data), len(self.img2))

            self.data = self.data[:cutoff]
            self.img2 = self.img2[:cutoff]

            self.min_length = cutoff
        else:
            self.min_length = len(self.data)

    def __getitem__(self, index):

        img_path = self.data[index]

        basename = os.path.basename(img_path)
        base = basename.split(".")[0]

        img_r = Image.open(img_path).convert("RGB")
        transform_params = get_params(self.opt, img_r.size)
        A_transform = get_transform(
            self.opt, transform_params, grayscale=(self.opt.input_nc == 1), norm=False
        )
        B_transform = get_transform(
            self.opt, transform_params, grayscale=(self.opt.output_nc == 1), norm=False
        )

        if self.mode != "train":
            A_transform = self.transform_r

        img_r = A_transform(img_r)

        B_mode = "L"
        if self.opt.output_nc == 3:
            B_mode = "RGB"

        img_depth = 0
        if self.midas:
            img_depth = cv2.imread(self.depth_maps[index])
            img_depth = A_transform(
                Image.fromarray(img_depth.astype(np.uint8)).convert("RGB")
            )

        img_normals = 0
        label = 0

        input_dict = {
            "r": img_r,
            "depth": img_depth,
            "path": img_path,
            "index": index,
            "name": base,
            "label": label,
        }

        if self.mode == "train":
            cur_path = self.img2[index]
            cur_img = B_transform(Image.open(cur_path).convert(B_mode))
            input_dict["line"] = cur_img

        return input_dict

    def __len__(self):
        return self.min_length


class VideoDataset(data.Dataset):
    def __init__(self, root, *args, transforms_r=None, **kwargs):
        video_files = make_dataset(root, filter_fn=is_video_file)
        num_frames_per_video = []
        self.videos = []
        for video_file in video_files:
            video = decord.VideoReader(video_file)
            self.videos.append(video)
            num_frames_per_video.append(len(video))
        self.transform_r = transforms.Compose(transforms_r)
        self.cumsum_frames = np.cumsum(num_frames_per_video)

    def __len__(self):
        return self.cumsum_frames[-1]

    def __getitem__(self, i):
        # Find the right video first
        video_index = bisect.bisect(self.cumsum_frames, i)
        # Find the right frame in the video
        if video_index > 0:
            frame_index = i - self.cumsum_frames[video_index - 1]
        else:
            frame_index = i
        frame = self.videos[video_index][frame_index]
        img_r = Image.fromarray(frame.asnumpy()).convert("RGB")
        img_r = self.transform_r(img_r)
        return {"r": img_r, "depth": 0, "name": f"frame_{i:d}"}
