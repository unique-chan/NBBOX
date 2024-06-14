<h1 align="center" style="font-weight: 500; line-height: 1.4;">
  NBBOX: Noisy Bounding Box Improves Remote Sensing Object Detection
</h1>

<p align="center">
  <a href="#"><img alt="Python3.7+" src="https://img.shields.io/badge/Python-3.7+-blue?logo=python&logoColor=white"></a>
  <a href="#"><img alt="PyTorch1.5+" src="https://img.shields.io/badge/PyTorch-1.5+-orange?logo=pytorch&logoColor=white"></a>
  <a href="#"><img alt="MMDetection2.28.2" src="https://img.shields.io/badge/MMDetection-2.28.2-red?logo=mmlab&logoColor=white"></a>
  <a href="#"><img alt="MMRotate0.3.4" src="https://img.shields.io/badge/MMRotate-0.3.4-hotpink?logo=mmlab&logoColor=white"></a>
  <a href="#"><img alt="MIT" src="https://img.shields.io/badge/License-MIT-green?logo=MIT"></a>
</p>

<p align="center">
  <b><a href="https://github.com/unique-chan">Yechan Kim</a></b>
</p>

### This repo includes:
- Official implementation of our proposed approach

### Overview:
- With our noisy bounding box transformation, you can boost remote sensing object detection.

### Preliminaries:
- Install all necessary packages listed in the `requirements.txt`. 
- Simply add our `NoisyBBOX` to *train_pipeline* in your model configuration file. Below is an example:
~~~python3
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='NoisyBBOX'), # Our transformation (⭐) should be placed directly after 'LoadAnnotations'
    ...
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
~~~

### Training via our strategy:
- Our `NoisyBBOX` has the following arguments to control:
  * `scale_range (min, max)` (dtype: float): Scale each bounding box by a random number between *min* and *max*, during training. Each box is affected by a different random number.
    * `isotropically_rescaled` (dtype: boolean): If this variable is `True`, each box is rescaled while preserving the aspect-ratio. Otherwise, not.
  * `angle_range (min, max)` (dtype: int): Rotate each bounding box by a random number between *min* and *max*, during training. Each box is affected by a different random number.
  * `translate_range (min, max)` (dtype: int): Translate each bounding box by a random number between *min* and *max*, during training. Each box is affected by a different random number.
    * `isotropically_translated` (dtype: boolean): If this variable is `True`, each box is translated while preserving the aspect-ratio. Otherwise, not.
- When training with this Git repo, you can easily control the arguments as follows:
  * Example: { scale_range: (0.7, 1.0), isotropically_rescaled: True }, angle_range: (-2, 2), { translate_range: (-1, 1), isotropically_translated: True }
  ~~~
  python train.py --model-config "my_src/my_cfg/rotated_faster_rcnn_r50_fpn_1x_dior_le90.py" \
                  --data-config "my_src/my_cfg/for_dior/cfg_for_data_DIOR_obb.py" \
                  --train-config "my_src/my_cfg/for_dior/cfg_for_train_DIOR_obb.py" \
                  --dataset-class "my_src/my_dataset/dior_obb.py" --batch-size 8   \
                  --data-root "DIOR" --epochs 1 --work-dir "work_dirs/dior_faster_r50" --gpu-id 0 --obb --tag "noisy-bbox" --init_weights \
                  --scale_min 0.7 --scale_max 1.0 --isotropically_rescaled \
                  --angle_min -2 --angle_max 2 \
                  --translate_min -1 --translate_max 1 --isotropically_translated // ⭐
  ~~~
- If you hope to train detectors with your own MMDetection/Rotate-based codes, please follow steps below:
  - Copy and paste `my_transform/noisy_bbox.py` into your code. Ensure that our transform to be registered in MMDetection/Rotate's PIPELINES.
  - Also, add our transform with parameters to *train_pipeline* in your model configuration file as follows:
  ~~~python3
  train_pipeline = [
    ...
    dict(type='NoisyBBOX', scale_range=(0.7, 1.0), 
         isotropically_rescaled=True, angle_range=(-2, 2)),
    ...
  ]
  ~~~


### Announcement:
- Code under construction
