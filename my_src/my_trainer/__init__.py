import importlib

from .train_for_common import *
from .train_for_hbb import *
from .train_for_obb import *
from .runner import *


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


def init_for_dynamic_backbone_freezing(args):
    '''
    Preliminary: Modify your detection model (or its superclass) as follows ->
        1) declare an attribute named `bool_freeze_backbone` (boolean variable)
        2) modify functions for a feature extractor (backbone) to be locked/unlocked by `bool_freeze_backbone`
        E.g. with MMDetection
        class SingleStageDetector(BaseDetector):  # mmdetection/mmdet/models/detectors/single_stage.py
            def __init__(...):
             self.bool_freeze_backbone = False
             ...
            def extract_feat(...):
             x = self.backbone(img)
             if self.with_neck:
                if self.bool_freeze_backbone:  x = self.neck(tuple([_.detach() for _ in x]))
                else:                          x = self.neck(x)
             else:
                if self.bool_freeze_backbone:  x = x.detach()
             return x
             ...
    '''
    if args.dbf is not None:
        fn_for_dynamic_backbone_freezing = (importlib.import_module(args.dbf.
                                                                    replace("/", ".").replace(".py", "")).
                                            dynamic_backbone_freezing)
        param_for_dynamic_backbone_freezing = eval(args.dbf_options)
        EpochBasedRunnerForDBF.fn_for_dynamic_backbone_freezing = fn_for_dynamic_backbone_freezing
        EpochBasedRunnerForDBF.param_for_dynamic_backbone_freezing = param_for_dynamic_backbone_freezing
