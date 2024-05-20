def get_config(cfg, args):
    num_classes = len(('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                       'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                       'basketball-court', 'storage-tank', 'soccer-ball-field',
                       'roundabout', 'harbor', 'swimming-pool', 'helicopter'))
    cfg.dataset_type = 'DOTAv1OBB'
    cfg.data_root = args.data_root
    # train
    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.data_root = f'{cfg.data_root}/train'
    cfg.data.train.ann_file = 'annfiles'
    cfg.data.train.img_prefix = 'images'

    # val
    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = f'{cfg.data_root}/val'
    cfg.data.val.ann_file = 'annfiles'
    cfg.data.val.img_prefix = 'images'

    # test
    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.data_root = f'{cfg.data_root}/test'
    cfg.data.test.ann_file = 'annfiles'
    cfg.data.test.img_prefix = 'images'
    return cfg
