import argparse

from mmcv import DictAction


class Parser:
    def __init__(self, mode):
        assert mode in ['train'], f'Unsupported mode: {mode} for Parser'
        self.parser = argparse.ArgumentParser()
        self.add_common_arguments()
        if mode == 'train':
            self.add_train_arguments()
            self.add_DBF_arguments()        # for dynamic backbone freezing

    def add_common_arguments(self):
        self.parser.add_argument('--model-config',
                                 help='model config file path, '
                                      'e.g. "mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"')
        self.parser.add_argument('--data-config',
                                 help='user-customized data-relevant config file path')
        self.parser.add_argument('--data-root', default='.', help='data root path')
        self.parser.add_argument('--work-dir', help='dir to save file containing eval metrics')
        self.parser.add_argument('--device', default='cuda',
                                 type=str, choices=['cpu', 'cuda'], help='"cpu" or "cuda"? (default: "cuda")')
        self.parser.add_argument('--gpu-id', type=int, default=0, nargs='+', help='id(s) of gpu(s) to use')
        self.parser.add_argument('--batch-size', type=int, help='batch size')
        self.parser.add_argument('--dataset-class',
                                 help='custom dataset class path for registration, '
                                      'e.g. "my_src/my_dataset/amod_v1_hbb.py"')
        self.parser.add_argument('--hbb', action='store_true')
        self.parser.add_argument('--obb', action='store_true')

    def add_train_arguments(self):
        self.parser.add_argument('--train-config',
                                 help='user-customized training-relevant config file path')
        self.parser.add_argument('--epochs', type=int, help='training_epochs')
        self.parser.add_argument('--load-from', help='checkpoint file (weights only)')
        self.parser.add_argument('--resume-from', help='checkpoint file to resume from')
        self.parser.add_argument('--no-validate', action='store_true',
                                 help='whether not to evaluate the checkpoint during training')
        self.parser.add_argument('--seed', default=0, type=int, help='random seed (default: 0)')
        self.parser.add_argument('--deterministic', action='store_true',
                                 help='whether to set deterministic options for CUDNN backend')
        self.parser.add_argument('--init_weights', action='store_true',
                                 help='use init weights')
        self.parser.add_argument('--tag', help='experiment tag')

    def add_DBF_arguments(self):
        self.parser.add_argument('--dbf', help='file for dynamic backbone freezing')
        self.parser.add_argument('--dbf-options',
                                 help='scheduling options for dynamic backbone freezing '
                                      '▶ e.g. %s' % '{"step_epoch": 10}')

    def parse_args(self):
        parsed_args = self.parser.parse_args()
        assert parsed_args.hbb != parsed_args.obb       # only one of them is True
        return parsed_args
