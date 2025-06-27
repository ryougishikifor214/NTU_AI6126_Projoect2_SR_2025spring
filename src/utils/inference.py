from mmagic.apis import MMagicInferencer, init_model
from pathlib import Path
import os
import os.path as osp
import shutil
import torch

from tqdm import tqdm

from project_config import ASSETS_CHECKPOINT_DIR_PATH

def inference_dir(img_dir: str, res_dir: str):
    if not osp.exists(res_dir):
        raise FileNotFoundError(f"Result directory {res_dir} does not exist.")
    
    if not osp.exists(img_dir):
        raise FileNotFoundError(f"Image directory {img_dir} does not exist.")
    
    model_name = res_dir.split('/')[-1]
    res_type = img_dir.split('/')[-1]
    
    config_file = osp.join(ASSETS_CHECKPOINT_DIR_PATH, model_name, 'config.py')
    checkpoint_file = osp.join(ASSETS_CHECKPOINT_DIR_PATH, model_name, 'checkpoint.pth')
    
    if not osp.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} does not exist.")
    if not osp.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_file} does not exist.")
    
    final_res_dir = osp.join(res_dir, res_type)
    if osp.exists(final_res_dir):
        shutil.rmtree(final_res_dir)
    os.makedirs(final_res_dir, exist_ok=False)
    
    inferencer = MMagicInferencer(
        model_name=model_name,
        model_config=config_file,
        model_ckpt=checkpoint_file,
        device= torch.device('cuda:0'),
    )
    model = init_model(
        config_file,
        checkpoint_file,
        device='cuda:0',
    )
    
    print(f"#Params of {config_file} and {checkpoint_file}: ", sum(p.numel() for p in model.parameters()))
    
    for img_file in tqdm([f for f in os.listdir(img_dir) if f.endswith('.png')]):
        img_file = osp.join(img_dir, img_file)
        inference_single_img(inferencer, img_file, final_res_dir)

def inference_single_img(inferencer: MMagicInferencer, img_file: str, final_res_dir: str):
    img_path = Path(img_file)
    if img_path.exists():
        img_name = img_path.name
        res_img_file = osp.join(final_res_dir, img_name)
        inferencer.infer(img=img_file, result_out_dir=res_img_file)
    else:
        raise FileNotFoundError(f"Image file {img_file} does not exist.")