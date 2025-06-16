# Bringing a Personal Point of View: Evaluating Dynamic 3D Gaussian Splatting for Egocentric Scene Reconstruction

This is the official implementation of my Master thesis titled *Bringing a Personal Point of View: Evaluating Dynamic 3D Gaussian Splatting for Egocentric Scene Reconstruction*. 

This repository is accompanied by the following modified implementation of models:
 - Deformable-3DGS: [https://github.com/Jaswar/def3dgs-thesis](https://github.com/Jaswar/def3dgs-thesis)
 - 4DGS: [https://github.com/Jaswar/4DGaussians-thesis](https://github.com/Jaswar/4DGaussians-thesis)
 - RTGS: [https://github.com/Jaswar/rtgs-thesis](https://github.com/Jaswar/rtgs-thesis)
 - EgoGaussian: [https://github.com/Jaswar/EgoGaussian-thesis](https://github.com/Jaswar/EgoGaussian-thesis)

Furthermore, the following repository contains a slighly-modified Segment Anything 2 implementation:
 - SAM2: [https://github.com/Jaswar/sam2-thesis](https://github.com/Jaswar/sam2-thesis)

All credits for original implementations go to the original authors. 

## Setting up

For information on how to setup the models, please refer to the corresponding repositories. This repository can be setup by installing the `requirements.txt` file. 

## Data

The dynamic masks can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1L7IHNWNZIseZ7K1Jagdp2EyK0z6UncVR?usp=sharing).

As per the licence of EgoExo4D, no part of the dataset is shared as part of this project. The takes as outlined in `settings.json` need to be downloaded from EgoExo4D directly. The script `prepare_ego_exo4d.py` must then be run to convert the data into expected format. Make sure to process the data inside the folder containing dynamic masks. 

## Attributions

The file `colmap_loader.py` comes from the original 3D Gaussian Splatting repository under the [Gaussian-Splatting licence](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md).

The file `common.py` contains fragments of masked-SSIM computation from [pytorch-mssim](https://github.com/VainF/pytorch-msssim/tree/master) under the [MIT license](https://github.com/VainF/pytorch-msssim/blob/master/LICENSE).
