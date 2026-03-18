from .bdf_model import BDFModel
from .unos_model import UnOSModel


def get_stereo_model(name, config):
    '''
    Model registry that maps a model name to the corresponding wrapper class.

    Arg(s):
        name : str
            model name ('bdf' or 'unos')
        config : dict
            model-specific configuration arguments

    Returns:
        BDFModel or UnOSModel : instantiated model wrapper
    '''

    name = name.lower()

    if name == 'bdf':
        return BDFModel(
            model_name=config.get('model_name', 'monodepth'),
            input_height=config.get('input_height', 256),
            input_width=config.get('input_width', 512),
            lr_loss_weight=config.get('lr_loss_weight', 0.5),
            alpha_image_loss=config.get('alpha_image_loss', 0.85),
            disp_gradient_loss_weight=config.get('disp_gradient_loss_weight', 0.1),
            type_of_2warp=config.get('type_of_2warp', 0),
            device=config.get('device', None),
        )
    elif name == 'unos':
        return UnOSModel(
            mode=config.get('mode', 'depthflow'),
            img_height=config.get('img_height', 256),
            img_width=config.get('img_width', 832),
            depth_smooth_weight=config.get('depth_smooth_weight', 10.0),
            ssim_weight=config.get('ssim_weight', 0.85),
            flow_smooth_weight=config.get('flow_smooth_weight', 10.0),
            flow_consist_weight=config.get('flow_consist_weight', 0.01),
            flow_diff_threshold=config.get('flow_diff_threshold', 4.0),
            num_scales=config.get('num_scales', 4),
            device=config.get('device', None),
        )
    else:
        raise ValueError(
            'Unknown stereo model: "{}". Supported models: "bdf", "unos".'.format(name))
