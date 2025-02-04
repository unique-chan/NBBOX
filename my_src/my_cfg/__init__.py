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
    warning_flag = True
    for i in range(len(code)):
        if "dict(type='NoisyBBOX'" in code[i]:
            print(f'* Original: {code[i]}', end='')
            code[i] = code[i].replace("'NoisyBBOX'",
                                      f"'NoisyBBOX', scale_range=({args.scale_min}, {args.scale_max}), "
                                      f"isotropically_rescaled={args.isotropically_rescaled}, "
                                      f"angle_range=({args.angle_min}, {args.angle_max}), "
                                      f"translate_range=({args.translate_min}, {args.translate_max}), "
                                      f"isotropically_translated={args.isotropically_translated}, "
                                      f"threshold={args.threshold}")
            print(f'* Modified: {code[i]}', end='')
            warning_flag = False
    f.close()

    if warning_flag:
        print('* No NoisyBBOX is found in the given config files!!!')

    # update model config for noisy bbox
    tmp_config = f'{os.path.dirname(args.model_config)}/{datetime.now().strftime("%Y%m%d_%H%M%S")}-model_config.py'
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
    n_bbox_params = []
    if args.scale_min != 1.0 or args.scale_max != 1.0:
        n_bbox_params.append(f'scale_min={args.scale_min}')
        n_bbox_params.append(f'scale_max={args.scale_max}')
        n_bbox_params.append(f'isotropically_rescaled={args.isotropically_rescaled}')

    if args.angle_min != 0 or args.angle_max != 0:
        n_bbox_params.append(f'angle_min={args.angle_min}')
        n_bbox_params.append(f'angle_max={args.angle_max}')

    if args.translate_min != 0 or args.translate_max != 0:
        n_bbox_params.append(f'translate_min={args.translate_min}')
        n_bbox_params.append(f'translate_max={args.translate_max}')
        n_bbox_params.append(f'isotropically_translated={args.isotropically_translated}')

    if args.threshold > 0:
        n_bbox_params.append(f'threshold={args.threshold}')

    _ = ['exp',
         args.train_config.split('/')[-1].replace('/', '.').replace('.py', ''),
         args.data_config.split('/')[-1].replace('/', '.').replace('.py', ''),
         '-'.join(n_bbox_params),
         datetime.now().strftime("%Y%m%d_%H%M%S")]
    if args.tag:
        _.insert(-1, args.tag)
    return f"{args.work_dir}/{'-'.join(_)}"
