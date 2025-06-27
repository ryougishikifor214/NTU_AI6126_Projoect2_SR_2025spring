## Introduction

All the checkpoints and report can be found in the following link
https://drive.google.com/drive/folders/1_u2jH5ymE5_VvZRqruLw1man53RfBAEU?usp=drive_link

## 1
matriculation No: G2303498k

CodaLab username: G2303498K

## 2: Description of files

**suggested conda environment: python3.8.20** and use `src/environment.ipynb` to install following required packages.

```
AI6126project2/
│
├── src/
|   |── project_config.py [define settings of project. OUT_xxx stores training logs and checkpoints. RES_xxx stores inference results]
│   ├── inference.ipynb [generate inference results]
│   |── evironment.ipynb [helps to install packages under selected conda environment]
|   ├── models/ [run following jupternotebooks to create corresponding training cofings]
|       ├── edsr.ipynb
|       ├── swinir.ipynb
|       ├── srgan.ipynb
│
├── mmagic/ [v1.20dev, modified by myself to add some functions compared to github version. Github version should work well also.]
|
│
|── assets/
|   ├── config/ [store training configs generated from src/models]
|   └── checkpoint/ [store final inference configs and checkpoints]
|
|── train.sh
```

## 3. third-party libraries
mmagic, 
mmcv, 
mmengine, 
opencv-python, 
pytorch2.10+cu118, 
albumentations, 

## 4. How to run

1. **Install conda environment**: use `src/environment.ipynb` to install required packages under selected conda environment.
2. **prepare dataset**: put dataset into correct places set by a series of `DATASETS_XXX` in `src/project_config.py`.
3. **run inference**: run `src/inference.ipynb` to generate inference results. The inference results will load checkpoints and configs from `assets/checkpoint` folder. The results will be saved under `ASSETS_XXX` in `src/project_config.py`.

**`RES_XXX`, `DATASET_XXX`should be modified according to your own settings. The default settings can't not work across different machines.**
