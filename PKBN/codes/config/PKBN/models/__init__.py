import logging

logger = logging.getLogger("base")


def create_model(opt):
    model = opt["model"]

    if model == "sr":
        from .SR_model import SRModel as M
    elif model == "srgan":
        from .SRGAN_model import SRGANModel as M
    elif model == "blind":
        from .blind_model import B_Model as M
    elif model == "blind_ker":
        from .blind_model_ker import B_Model as M
    elif model == "SRDiff":
        from .SRDiff_model import B_Model as M
    else:
        raise NotImplementedError("Model [{:s}] not recognized.".format(model))
    m = M(opt)
    logger.info("Model [{:s}] is created.".format(m.__class__.__name__))
    return m
