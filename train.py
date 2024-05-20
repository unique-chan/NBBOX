# from mmdet.datasets import build_dataset -> already imported in my_src/my_trainer/train_for_hbb.py
# from mmdet.apis import set_random_seed   -> already imported in my_src/my_trainer/train_for_hbb.py
# from mmdet.apis import train_detector    -> replaced with train_detector() in my_src/my_trainer/train_for_hbb.py

from mmdet.models import build_detector as bd_hbb
from mmrotate.models import build_detector as bd_obb

from my_src import *

if __name__ == '__main__':
    args = Parser('train').parse_args()

    set_random_seed(args.seed, deterministic=args.deterministic)
    cfg = get_all_configs(args, mode='train', verbose=False)

    datasets = [build_dataset(cfg.data.train)]
    _build_detector = bd_hbb if args.hbb else bd_obb
    model = _build_detector(cfg.model,
                            train_cfg=cfg.get('train_cfg'),
                            test_cfg=cfg.get('test_cfg'))
    if args.init_weights:
        model.init_weights()
    model.CLASSES = datasets[0].CLASSES

    init_workdir_and_cfg_dump(cfg, args)
    init_for_dynamic_backbone_freezing(args)
    _train_detector = train_detector_for_hbb if args.hbb else train_detector_for_obb
    runner = _train_detector(model, datasets, cfg, distributed=False,
                             validate=(not args.no_validate), run_time_measure=True)
    save_log_from_runner(cfg.work_dir, runner)
