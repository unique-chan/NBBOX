def get_config(cfg, args):
    num_classes = len(('Armored', 'Artillery', 'Boat', 'Helicopter', 'LCU', 'MLRS', 'Plane', 'RADAR', 'SAM',
                       'Self-propelled Artillery', 'Support', 'TEL', 'Tank'))

    cfg.dataset_type = 'AMODv1HBB'
    cfg.data_root = args.data_root
    # train
    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.data_root = f'{cfg.data_root}/train'
    cfg.data.train.ann_file = ''
    cfg.data.train.img_prefix = ''
    cfg.data.train.pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(640, 480), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', mean=[100, 100, 100], std=[50, 50, 50], to_rgb=True),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ]
    # val
    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = f'{cfg.data_root}/val'
    cfg.data.val.ann_file = ''
    cfg.data.val.img_prefix = ''
    cfg.data.val.pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(640, 480),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', mean=[100, 100, 100], std=[50, 50, 50], to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])
    ]
    # test
    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.data_root = f'{cfg.data_root}/test'
    cfg.data.test.ann_file = ''
    cfg.data.test.img_prefix = ''
    cfg.data.test.pipeline = cfg.data.val.pipeline
    # set number of classes for head
    try:
        cfg.model.bbox_head.num_classes = num_classes
    except:
        pass
    try:
        cfg.model.roi_head.bbox_head.num_classes = num_classes
    except:
        pass
    try:
        for _ in cfg.model.roi_head.bbox_head:
            _['num_classes'] = num_classes
    except:
        pass
    return cfg
