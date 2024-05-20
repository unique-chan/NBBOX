def dynamic_backbone_freezing(runner):
    model = runner.model.module
    epoch = runner._epoch
    param = runner.param_for_dynamic_backbone_freezing
    assert (type(param) is dict
            and 'steps' in param.keys()), 'DBF_ARGS needed -> e.g. "%s"' % '{"steps": [1,10,20]}'
    if epoch in param['steps']:
        model.bool_freeze_backbone = False  # unlock backbone
    else:
        model.bool_freeze_backbone = True   # lock backbone
