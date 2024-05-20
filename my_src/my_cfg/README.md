Please follow the below convention to name config files:
~~~
{model}_[model setting]_{backbone}_{neck}_[norm setting]_[misc]_[gpu x batch_per_gpu]_{schedule}_{dataset}.py
~~~

{xxx} is required field and [yyy] is optional.

* `{model}`: model type like faster_rcnn, mask_rcnn, etc.
* `[model setting]`: specific setting for some model, like without_semantic for htc, moment for reppoints, etc.
* `{backbone}`: backbone type like r50 (ResNet-50), x101 (ResNeXt-101).
* `{neck}`: neck type like fpn, pafpn, nasfpn, c4.
* `[norm_setting]`: bn (Batch Normalization) is used unless specified, other norm layer type could be gn (Group Normalization), syncbn (Synchronized Batch Normalization). gn-head/gn-neck indicates GN is applied in head/neck only, while gn-all means GN is applied in the entire model, e.g. backbone, neck, head.
* `[misc]`: miscellaneous setting/plugins of model, e.g. dconv, gcb, attention, albu, mstrain.
* `[gpu x batch_per_gpu]`: GPUs and samples per GPU, 8x2 is used by default.
* `{schedule}`: training schedule, options are 1x, 2x, 20e, etc. 1x and 2x means 12 epochs and 24 epochs respectively. 20e is adopted in cascade models, which denotes 20 epochs. For 1x/2x, initial learning rate decays by a factor of 10 at the 8/16th and 11/22th epochs. For 20e, initial learning rate decays by a factor of 10 at the 16th and 19th epochs.
* `{dataset}`: dataset like coco, cityscapes, voc_0712, wider_face.

Please refer to https://mmdetection.readthedocs.io/en/v2.28.2/tutorials/config.html#config-name-style for details.