from enum import Enum, auto

class VisMode(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()

class VisBackendType(Enum):
    LOCAL = auto()
    WANDB = auto()
    ALL = auto()

wandb_metric_cfgs = [
    dict(name='lr', step_metric='iter'),
    dict(name='PSNR', step_metric='iter', step_sync=True),
    dict(name='MAE', step_metric='iter', step_sync=True),
    dict(name='SSIM', step_metric='iter', step_sync=True),
]
wandb_visbackend = dict(
    type = 'WandbVisBackend',
    init_kwargs = dict(
        project = 'AI6126project2',
        name = '', # TODO in models/xxxx.ipynb, name of the experiment
        resume = 'allow',
        allow_val_change = True,
    ),
    define_metric_cfg = wandb_metric_cfgs,
)
local_visbackend = dict(type='LocalVisBackend')

def get_vis_backends(vis_type: VisBackendType = VisBackendType.WANDB):
    vis_backends = []
    if vis_type == VisBackendType.LOCAL:
        vis_backends.append(local_visbackend)
    elif vis_type == VisBackendType.WANDB:
        vis_backends.append(wandb_visbackend)
    elif vis_type == VisBackendType.ALL:
        vis_backends.append(local_visbackend)
        vis_backends.append(wandb_visbackend)
    else:
        raise ValueError(f"Invalid vis backend type: {vis_type}")
    return vis_backends

    
def get_visualizer_and_custom_hook(vis_mode: VisMode = VisMode.VAL, vis_type: VisBackendType = VisBackendType.WANDB):
    vis_backends = get_vis_backends(vis_type)
    if vis_mode == VisMode.TRAIN:
        visualizer = dict(
            type='ConcatImageVisualizer',
            vis_backends=vis_backends,
            fn_key='gt_path',
            img_keys=['gt_img', 'input'],
            bgr2rgb=False,
        )
        vis_hook = dict(type='BasicVisualizationHook', 
                    on_train = True,
                    on_val = False,
                    on_test = False,
                    interval = 100,
        )
    elif vis_mode in [VisMode.VAL, VisMode.TEST]:
        visualizer = dict(
            type='ConcatImageVisualizer',
            vis_backends=vis_backends,
            fn_key='gt_path',
            img_keys=['gt_img', 'input', 'pred_img'],
            bgr2rgb=True,
        )
        vis_hook = dict(type='BasicVisualizationHook', 
                    on_train = False,
                    on_val = True,
                    on_test = True,
                    interval = 1,
        )
    else:
        raise ValueError(f"Invalid vismode: {vis_mode}")
    return visualizer, vis_hook


class ParamSchedulerType(Enum):
    MULTISTEP = auto()
    COSINE = auto()


def get_param_scheduler(scheduler_type: ParamSchedulerType, kwargs: dict):
    if scheduler_type == ParamSchedulerType.MULTISTEP:
        param_scheduler = dict(
            type='MultiStepLR',
            by_epoch = kwargs.get('by_epoch', False),
            gamma = kwargs.get('gamma', 0.5),
            milestones = kwargs['milestones'],
        )
    elif scheduler_type == ParamSchedulerType.COSINE:
        param_scheduler = dict(
            type='CosineRestartLR',
            by_epoch = kwargs.get('by_epoch', False),
            periods = kwargs['periods'],
            restart_weights = kwargs.get('restart_weights', [1]*len(kwargs['periods'])),
            eta_min = kwargs['eta_min']
        )
    else:
        raise ValueError(f"Invalid param scheduler type: {scheduler_type}")
    return param_scheduler


DA_TYPES = {
    'PairedRandomCrop': 'PRC',
    'Flip': 'Flip',
    'RandomRotationWithRatio': 'RR',
    'Cutblur': 'Cutblur',
    'RandomTransposeHW': 'RT',
}

def get_DA_repr(cfg) -> str:
    da_types = []
    for transform in cfg.train_pipeline:
        if transform.type in DA_TYPES.keys():
            if transform.type == 'Flip':
                if transform.direction == 'horizontal':
                    da_types.append('FlipH')
                elif transform.direction == 'vertical':
                    da_types.append('FlipV')
                else:
                    raise ValueError(f"Invalid flip direction: {transform.direction}")
            else:
                da_types.append(DA_TYPES[transform.type])    
    return '_'.join(da_types) if len(da_types) else ''