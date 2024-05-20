import importlib
from datetime import datetime

from mmcv import Config

from .parser import *


def get_config(cfg_file, cfg, args):
    # Your custom config file should be a python file
    # including a function named "get_config(cfg, args)" where cfg & args: param!
    current_module = importlib.import_module(cfg_file.replace("/", ".").replace(".py", ""))
    return current_module.get_config(cfg, args)


def get_all_configs(args, mode='train', verbose=True):
    cfg = Config.fromfile(args.model_config)
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
         datetime.now().strftime("%Y%m%d_%H%M%S")]
    if args.dbf:
        _.insert(-1, f"{args.dbf.split('/')[-1].replace('/', '.').replace('.py', '')}-{args.dbf_options}")
        _[-2] = _[-2].replace('"', '').replace("'", '').replace(':', '')
    if args.tag:
        _.insert(-1, args.tag)
    return f"{args.work_dir}/{'-'.join(_)}"
