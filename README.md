# Informative Drawings: Learning to generate line drawings that convey geometry and semantics

### [[project page]](https://carolineec.github.io/informative_drawings/) [[paper]](https://arxiv.org/abs/2203.12691) [[video]](TODO) [[demo]](https://huggingface.co/spaces/carolineec/informativedrawings)


## Setup

### Clone this repository

```
git clone https://github.com/carolineec/informative-drawings.git
cd informative-drawings
```

### Install dependencies
We provide an environment.yml file listing the dependences and to create a conda environment. Our model uses Pytorch 1.7.1

```
conda env create -f environment.yml
conda activate drawings
```

Use the following command to install [CLIP](https://github.com/openai/CLIP) (only needed for training).

```
conda activate drawings
pip install git+https://github.com/openai/CLIP.git
```

## Testing
Pre-trained model is available [here](https://drive.google.com/file/d/11l5u5sb1PO5Z5YA3IoEHauVPm0k407C1/view?usp=sharing), place the model weights in `checkpoints`.

```
cd checkpoints
unzip model.zip
```

run pre-trained model on images in `--dataroot`. Replace `examples ` with the folder path containing your input images.

```
python test.py --name anime_style --dataroot examples/test
```

Results will be saved to the `results` directory by default. You can change the save location by specifying the file path with `--results_dir`. 

## Training

We provide a pre-trained network for mapping ImageNet features into depth images [here](https://drive.google.com/file/d/1aEi-IdT1qq1gDCfEn-jd7jxhC6fEW-OE/view?usp=sharing). Place the pre-trained features to depth network in the `./checkpoints/feats2Geom` folder.

```
cd checkpoints/feats2Geom
unzip feats2depth.zip
```

To train a model with name `myexperiment` from scratch use the following command.

```
python train.py --name myexperiment \
--dataroot examples/train/photos \
--depthroot examples/train/depthmaps \
--root2 examples/train/drawings \
--no_flip
```
Replace the example data `examples/train/photos`, `examples/train/depthmaps`, and `examples/train/drawings` with the paths to the dataset of photographs, depth maps, and line drawings respectively. Corresponding images and depth maps in the file paths specified by `--dataroot` and `--depthroot` should have the same file names. You will also need to specify a path to an unaligned dataset of line drawings with `--root2`. A small example of training data is provided in `examples/train`.

Because the model can start making grayscale photos after some training, it is recommended to save model checkpoints frequently by adding the flag `--save_epoch_freq 1`.

### Depth Maps

For training, geometry supervision requires depth maps for the dataset of photographs. To produce psuedo-ground truth depth maps we rely on a pretrained model from [Boosting Monocular Depth Estimation Models to High-Resolution via Content-Adaptive Multi-Resolution Merging](http://yaksoy.github.io/highresdepth/).

## Citation

If you find this work useful please use the following citation:

```
 @article{chan2022drawings,
	      title={Learning to generate line drawings that convey geometry and semantics},
	      author={Chan, Caroline and Durand, Fredo and Isola, Phillip},
	      booktitle={CVPR},
	      year={2022}
	      }
```

## Acknowledgements

Model code adapted from [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) and [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
