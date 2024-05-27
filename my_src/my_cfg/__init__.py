import importlib
import random
import os
import time
from datetime import datetime

from mmcv import Config

from .parser import *


def get_config(cfg_file, cfg, args):
    # Your custom config file should be a python file
    # including a function named "get_config(cfg, args)" where cfg & args: param!
    current_module = importlib.import_module(cfg_file.replace("/", ".").replace(".py", ""))
    return current_module.get_config(cfg, args)


def get_all_configs(args, mode='train', verbose=True):
    # read original model config
    f = open(args.model_config, 'r')
    code = f.readlines()
    for i in range(len(code)):
        if "dict(type='NoisyBBOX'" in code[i]:
            print(f'* Original: {code[i]}', end='')
            code[i] = code[i].replace("'NoisyBBOX'",
                                      f"'NoisyBBOX', scale_range=({args.scale_min}, {args.scale_max}), "
                                      f"isotropically_rescaled={args.isotropically_rescaled}, "
                                      f"angle_range=({args.angle_min}, {args.angle_max}) ")
            print(f'* Modified: {code[i]}', end='')
    f.close()

    # update model config for noisy bbox
    tmp_config = f'my_src/my_cfg/{datetime.now().strftime("%Y%m%d_%H%M%S")}-model_config.py'
    f = open(tmp_config, 'w')
    f.writelines(code)
    f.close()

    # load updated model config file to mmdetection/mmrotate
    cfg = Config.fromfile(tmp_config)

    # remove updated model config file
    time.sleep(0.5)
    os.remove(tmp_config)

    cfg = get_config(args.data_config, cfg, args)

    cfg.device = args.device
    cfg.gpu_ids = args.gpu_id   # if device == 'cuda', gpu_id should be specified!
    if mode == 'train':
        cfg.seed = args.seed
        cfg.load_from = args.load_from
        cfg.resume_from = args.resume_from
        cfg.runner.max_epochs = args.epochs
        cfg.work_dir = get_work_dir(args)
        cfg = get_config(args.train_config, cfg, args)
    if verbose:
        print(f'▶️ {cfg.pretty_text}')
    return cfg


def get_work_dir(args):
    _ = ['exp',
         args.train_config.split('/')[-1].replace('/', '.').replace('.py', ''),
         args.data_config.split('/')[-1].replace('/', '.').replace('.py', ''),
         f's_min={args.scale_min}-s_max={args.scale_max}-ir={args.isotropically_rescaled}-'
         f'a_min={args.angle_min}-a_max={args.angle_max}',
         datetime.now().strftime("%Y%m%d_%H%M%S")]
    if args.tag:
        _.insert(-1, args.tag)
    return f"{args.work_dir}/{'-'.join(_)}"
