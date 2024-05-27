import importlib

from .train_for_common import *
from .train_for_hbb import *
from .train_for_obb import *


def _object_to_txt(txt, txt_file_path):
    with open(txt_file_path, 'w') as fp:
        if type(txt) is not str:
            txt = str(txt)
        fp.write(txt)


def save_log_from_runner(work_dir, runner):
    _object_to_txt(runner.meta, f'{work_dir}/runner.meta.txt')
    _object_to_txt(runner.outputs, f'{work_dir}/runner.outputs.txt')
    _object_to_txt('', f'{work_dir}/best_mAP_val_{runner.meta["hook_msgs"]["best_score"]}')
    if runner.meta.get('run_time'):
        _object_to_txt(runner.meta, f'{work_dir}/runtime_{runner.meta.get("run_time")}')
