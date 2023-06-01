from ..utils import Registery

BACKBONE_REGISTERY = Registery("BACKBONE")

def build(rootmodel, cfg, name_in_cfg, alias=None):
    module = None
    if name_in_cfg in cfg:
        module_cfg = cfg.pop(name_in_cfg)
        module_type = module_cfg.pop("type")

        module = BACKBONE_REGISTERY.load(module_type)(**module_cfg)

        module_cfg['type'] = module_type
        cfg[name_in_cfg] = module_cfg

    if rootmodel is not None:
        if alias is not None:
            rootmodel.add_module(alias, module)
        else:
            rootmodel.add_module(name_in_cfg, module)