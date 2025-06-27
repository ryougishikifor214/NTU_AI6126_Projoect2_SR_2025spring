import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
from pathlib import Path
from typing import List

import torch
import gc

from mmengine import Config
from mmengine.logging import MMLogger
from mmengine.registry import init_default_scope

try:
    from mmengine.analysis import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')

from mmagic.registry import MODELS
logger = MMLogger.get_instance(name='MMLogger', log_level="ERROR")


def parse_args(inputs: List[str]):
    parser = argparse.ArgumentParser(
        description='Get a editor complexity',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[3, 250, 250],
        help='Input shape. Supported tasks:\n'
        'Image Super-Resolution: --shape 3 h w\n'
        'Video Super-Resolution: --shape t 3 h w\n'
        'Video Interpolation: --shape t 3 h w\n'
        'Image Restoration: --shape 3 h w\n'
        'Inpainting: --shape 4 h w\n'
        'Matting: --shape 4 h w\n'
        'Unconditional GANs: --shape noisy_size\n'
        'Image Translation: --shape 3 h w')
    parser.add_argument(
        '--activations',
        action='store_true',
        help='Whether to show the Activations')
    parser.add_argument(
        '--out-table',
        action='store_true',
        help='Whether to show the complexity table')
    parser.add_argument(
        '--out-arch',
        action='store_true',
        help='Whether to show the complexity arch')
    args = parser.parse_args(inputs)
    return args

def inference_nflops_nparams(inputs: List[str], logger: MMLogger = logger):
    args = parse_args(inputs)

    input_shape = tuple(args.shape)

    config_name = Path(args.config)
    if not config_name.exists():
        logger.error(f'Config file {config_name} does not exist')
    
    cfg = Config.fromfile(args.config)

    init_default_scope(cfg.get('default_scope', 'mmagic'))

    model = MODELS.build(cfg.model)
    inputs = torch.randn(1, *input_shape)
    if torch.cuda.is_available():
        model.cuda()
        inputs = inputs.cuda()
    model.eval()

    if hasattr(model, 'generator'):
        model = model.generator
    elif hasattr(model, 'backbone'):
        model = model.backbone
    if hasattr(model, 'translation'):
        model.forward = model.translation
    elif hasattr(model, 'infer'):
        model.forward = model.infer

    analysis_results = get_model_complexity_info(model, inputs=inputs)
    flops = analysis_results['flops_str']
    params = analysis_results['params_str']
    activations = analysis_results['activations_str']

    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}\n')
    
    if args.activations:
        print(f'Activations: {activations}\n{split_line}\n')
    if args.out_table:
        print(analysis_results['out_table'], '\n')
    if args.out_arch:
        print(analysis_results['out_arch'], '\n')
    # if len(input_shape) == 4:
    #     print('!!!If your network computes N frames in one forward pass, you '
    #           'may want to divide the FLOPs by N to get the average FLOPs '
    #           'for each frame.')
    # print('!!!Please be cautious if you use the results in papers. '
    #       'You may need to check if all ops are supported and verify that the '
    #       'flops computation is correct.')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return flops, params

from collections import defaultdict
def to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        d = {k: to_dict(v) for k, v in d.items()}
    return d

units = {"K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12,}
def convert_size_to_int(size_str):
    """Convert a size string like '1.4M', '50K', '0.936G' to an integer in bytes."""
    size_str = size_str.strip().upper()
    
    if size_str[-1] in units:
        num = float(size_str[:-1])
        return int(num * units[size_str[-1]])
    else:
        return int(size_str)
    

def filter_dict_by_flag(d, key="flag", target_value=True):
    if isinstance(d, dict):
        new_dict = {}
        for k, v in d.items():
            filtered_value = filter_dict_by_flag(v, key, target_value)
            if isinstance(filtered_value, dict) and filtered_value:
                new_dict[k] = filtered_value
            elif isinstance(filtered_value, list) and filtered_value:
                new_dict[k] = filtered_value
            elif isinstance(v, dict) and v.get(key) == target_value:
                new_dict[k] = v
        return new_dict if new_dict else None
    elif isinstance(d, list): 
        filtered_list = [filter_dict_by_flag(item, key, target_value) for item in d]
        return [item for item in filtered_list if item]  
    else:
        return None
    
    
import pprint
def save_to_py(filepath, var_name, data):
    filepath = Path(filepath)
    with filepath.open("w", encoding="utf-8") as f:
        f.write(f"{var_name} = ")
        f.write(pprint.pformat(to_dict(data), indent=2))
        f.write("\n")
        
import multiprocessing as mp
import traceback
def run_inference_worker(queue, inputs):
    try:
        result = inference_nflops_nparams(inputs)
        queue.put(result)
    except Exception as e:
        traceback.print_exc()
        queue.put(e)

def run_model_in_subprocess(inputs):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(target=run_inference_worker, args=(queue, inputs))
    p.start()
    p.join()

    result = queue.get()
    if isinstance(result, Exception):
        raise result
    return result


