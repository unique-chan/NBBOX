def get_config(cfg, args):
    cfg.optimizer.lr = 0.001
    cfg.lr_config = dict(
        policy='step',
        warmup=None,
        warmup_iters=500,
        warmup_ratio=0.001,
        step=[100]
    )
    # cfg.workflow = [('train', 1), ('val', 1)]
    cfg.checkpoint_config.interval = -1  # save only when val mAP is best
    cfg.log_config.interval = 100
    cfg.log_config.hooks = [dict(type='TextLoggerHook'),
                            dict(type='TensorboardLoggerHook')]
    cfg.evaluation.metric = 'mAP'
    cfg.evaluation.save_best = 'mAP'
    cfg.evaluation.interval = 1
    return cfg
