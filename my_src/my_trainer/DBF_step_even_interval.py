def dynamic_backbone_freezing(runner):
    model = runner.model.module
    epoch = runner._epoch
    param = runner.param_for_dynamic_backbone_freezing
    assert (type(param) is dict
            and 'step_epoch' in param.keys()), 'DBF_ARGS needed -> e.g. "%s"' % '{"step_epoch": 10}'
    if epoch % param['step_epoch'] == 0:
        model.bool_freeze_backbone = False  # unlock backbone
    else:
        model.bool_freeze_backbone = True   # lock backbone
