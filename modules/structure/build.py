from ..utils import Registery

STRUCTURE_REGISTERY = Registery("STRUCTURE")

def build_model(cfg, mode='Model'):
    if 'Model' in cfg:
        mode = cfg['Model']['type']
    
    model = STRUCTURE_REGISTERY.load(mode)(cfg)
    return model