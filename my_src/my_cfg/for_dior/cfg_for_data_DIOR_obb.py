def get_config(cfg, args):
    XML_TYPE = 'obb'
    cfg.dataset_type = 'DIORDataset'
    cfg.data_root = args.data_root
    # train
    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.xmltype = XML_TYPE
    cfg.data.train.data_root = ''
    cfg.data.train.ann_file = 'DIOR/ImageSets/Main/train.txt'
    cfg.data.train.img_prefix = 'DIOR/JPEGImages-trainval'

    # val
    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.xmltype = XML_TYPE
    cfg.data.val_data_root = ''
    cfg.data.val.ann_file = 'DIOR/ImageSets/Main/val.txt'
    cfg.data.val.img_prefix = 'DIOR/JPEGImages-trainval'

    # test
    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.xmltype = XML_TYPE
    cfg.data.test.data_root = ''
    cfg.data.test.ann_file = 'DIOR/ImageSets/Main/test.txt'
    cfg.data.test.img_prefix = 'DIOR/JPEGImages-test'
    return cfg
