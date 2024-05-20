import os

import numpy as np
import pandas as pd

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset


@DATASETS.register_module(force=True)
class AMODv1HBB(CustomDataset):
    CLASSES = (
        'Armored', 'Artillery', 'Boat', 'Helicopter', 'LCU', 'MLRS', 'Plane', 'RADAR', 'SAM',
        'Self-propelled Artillery', 'Support', 'TEL', 'Tank'
    )

    def load_annotations(self, ann_file):
        """ (1) About dataset directory architecture
            📁 e.g. data_root
             |— 📁 0000                           # The current image (sample) index is 0000.
                 |— 📁 0                          # The current look angle is 0 degree.
                     |— 🖼️️ EO_0000_0.png          # The electro-optical image satisfying the above conditions is given.
                     |— 📄 ANNOTATION_0000_0.csv  # The annotation for the above image is given.
             |— 📁 0001
             |— ...

            (2) About annotation file structure
            📄 e.g. ANNOTATION_0000_0.csv
             |  main_class  | min_x     | min_y     | max_x     | max_y     |
             |——————————————+———————————+———————————+———————————+———————————|
             |   MLRS       | 289       | 393       | 321       | 444       |
             |   ...        | ...       | ...       | ...       | ...       |
        """
        sample_idx_list = os.listdir(self.data_root)  # list of all image indices e.g. ['0000', '0001', ...]
        data_info_list = []
        for sample_idx in sample_idx_list:
            annot_df = pd.read_csv(f'{self.data_root}/{sample_idx}/0/'
                                   f'ANNOTATION_{sample_idx}_0.csv').query('usable == "T"')
            data_info_list.append({
                'filename': f'{sample_idx}/{0}/EO_{sample_idx}_{0}.png',  # e.g. '0015/0/EO_0015_0.png'
                'width': 640, 'height': 480,
                'ann': {
                    'bboxes': np.array(annot_df[['min_x', 'min_y', 'max_x', 'max_y']], dtype=np.float32),
                    'labels': np.array(list(map(lambda label: self.CLASSES.index(label),
                                                # this lambda works as follows: e.g. 'Artillery' -> 1, 'LCU' -> 4
                                                list(annot_df['main_class']))), dtype=np.int64),
                }
            })
        return data_info_list
