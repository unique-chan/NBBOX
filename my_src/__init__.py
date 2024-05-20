from .my_cfg import *
from .my_dataset import *
from .my_trainer import *


def init_workdir_and_cfg_dump(cfg, args):
    os.makedirs(cfg.work_dir, exist_ok=True)
    cfg.dump(f'{cfg.work_dir}/cfg.py')
    if args.dataset_class:
        with open(f'{cfg.work_dir}/cfg.py', 'a') as f_out:
            with open(args.dataset_class) as f_in:
                dataset_class = f_in.read()
            f_out.write(dataset_class)
    else:
        print('⚠️ [Warning] No "dataset-class" given in the parser arguments!')
    with open(f'{cfg.work_dir}/args.txt', 'w') as f_out:
        f_out.write(str(args))
