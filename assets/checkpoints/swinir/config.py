DATASET_TRAIN_DIR_PATH = '/home/featurize/data/ffhq/train'
DATASET_VAL_DIR_PATH = '/home/featurize/data/ffhq/val'
custom_hooks = [
    dict(type='WandbValMetricHook'),
    dict(
        interval=1,
        on_test=True,
        on_train=False,
        on_val=True,
        type='BasicVisualizationHook'),
]
data_root = 'data'
dataset_type = 'BasicImageDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=5000,
        max_keep_ckpts=2,
        out_dir=
        '/home/featurize/out/AI6126project2/swinir/swinir_iter500000_psMULTISTEP_opAdam_daPRC_FlipH_FlipV_RT_nTrue',
        rule='greater',
        save_best='PSNR',
        type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmagic'
div2k_data_root = 'data/DIV2K'
div2k_dataloader = dict(
    dataset=dict(
        ann_file='meta_info_DIV2K100sub_GT.txt',
        data_prefix=dict(
            gt='DIV2K_train_HR_sub', img='DIV2K_train_LR_bicubic/X4_sub'),
        data_root='data/DIV2K',
        metainfo=dict(dataset_type='div2k', task_name='sisr'),
        pipeline=[
            dict(
                channel_order='rgb',
                color_type='color',
                imdecode_backend='cv2',
                key='img',
                type='LoadImageFromFile'),
            dict(
                channel_order='rgb',
                color_type='color',
                imdecode_backend='cv2',
                key='gt',
                type='LoadImageFromFile'),
            dict(type='PackInputs'),
        ],
        type='BasicImageDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
div2k_evaluator = dict(
    metrics=[
        dict(crop_border=4, prefix='DIV2K', type='PSNR'),
        dict(crop_border=4, prefix='DIV2K', type='SSIM'),
    ],
    type='Evaluator')
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=4))
evaluator = dict(
    metrics=[
        dict(convert_to='Y', crop_border=4, prefix='DIV2K', type='PSNR'),
        dict(convert_to='Y', crop_border=4, prefix='DIV2K', type='SSIM'),
    ],
    type='Evaluator')
experiment_name = 'swinir_iter500000_psMULTISTEP_opAdam_daPRC_FlipH_FlipV_RT_nTrue'
gt_h_size = 512
gt_w_size = 512
img_size = 64
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=100)
metric = dict(convert_to='Y', crop_border=4, prefix='DIV2K', type='SSIM')
model = dict(
    data_preprocessor=dict(
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='DataPreprocessor'),
    generator=dict(
        depths=[
            6,
            6,
            6,
            6,
        ],
        embed_dim=60,
        img_range=1.0,
        img_size=64,
        in_chans=3,
        mlp_ratio=2,
        num_heads=[
            6,
            6,
            6,
            6,
        ],
        resi_connection='1conv',
        type='SwinIRNet',
        upsampler='pixelshuffledirect',
        upscale=4,
        window_size=8),
    pixel_loss=dict(loss_weight=1.0, reduction='mean', type='L1Loss'),
    type='BaseEditModel')
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    optimizer=dict(betas=(
        0.9,
        0.999,
    ), lr=0.0002, type='Adam'),
    type='OptimWrapper')
param_scheduler = dict(
    by_epoch=False,
    gamma=0.5,
    milestones=[
        250000,
        400000,
        450000,
        475000,
    ],
    type='MultiStepLR')
resume = False
save_dir = '/home/featurize/out/AI6126project2/swinir/swinir_iter500000_psMULTISTEP_opAdam_daPRC_FlipH_FlipV_RT_nTrue'
scale = 4
set14_data_root = 'data/Set14'
set14_dataloader = dict(
    dataset=dict(
        data_prefix=dict(gt='GTmod12', img='LRbicx4'),
        data_root='data/Set14',
        metainfo=dict(dataset_type='set14', task_name='sisr'),
        pipeline=[
            dict(
                channel_order='rgb',
                color_type='color',
                imdecode_backend='cv2',
                key='img',
                type='LoadImageFromFile'),
            dict(
                channel_order='rgb',
                color_type='color',
                imdecode_backend='cv2',
                key='gt',
                type='LoadImageFromFile'),
            dict(type='PackInputs'),
        ],
        type='BasicImageDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
set14_evaluator = dict(
    metrics=[
        dict(crop_border=4, prefix='Set14', type='PSNR'),
        dict(crop_border=4, prefix='Set14', type='SSIM'),
    ],
    type='Evaluator')
set5_data_root = 'data/Set5'
set5_dataloader = dict(
    dataset=dict(
        data_prefix=dict(gt='GTmod12', img='LRbicx4'),
        data_root='data/Set5',
        metainfo=dict(dataset_type='set5', task_name='sisr'),
        pipeline=[
            dict(
                channel_order='rgb',
                color_type='color',
                imdecode_backend='cv2',
                key='img',
                type='LoadImageFromFile'),
            dict(
                channel_order='rgb',
                color_type='color',
                imdecode_backend='cv2',
                key='gt',
                type='LoadImageFromFile'),
            dict(type='PackInputs'),
        ],
        type='BasicImageDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
set5_evaluator = dict(
    metrics=[
        dict(crop_border=4, prefix='Set5', type='PSNR'),
        dict(crop_border=4, prefix='Set5', type='SSIM'),
    ],
    type='Evaluator')
test_cfg = dict(type='MultiTestLoop')
test_dataloader = dict(
    dataset=dict(
        data_prefix=dict(gt='GT', img='LQ'),
        data_root='/home/featurize/data/ffhq/val',
        filename_tmpl=dict(gt='{}', img='{}'),
        pipeline=[
            dict(
                channel_order='rgb',
                color_type='color',
                imdecode_backend='cv2',
                key='img',
                type='LoadImageFromFile'),
            dict(
                channel_order='rgb',
                color_type='color',
                imdecode_backend='cv2',
                key='gt',
                type='LoadImageFromFile'),
            dict(keys=[
                'img',
                'gt',
            ], type='PackInputs'),
        ],
        type='BasicImageDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(type='MAE'),
    dict(crop_border=4, type='PSNR'),
    dict(crop_border=4, type='SSIM'),
]
test_pipeline = [
    dict(
        channel_order='rgb',
        color_type='color',
        imdecode_backend='cv2',
        key='img',
        type='LoadImageFromFile'),
    dict(
        channel_order='rgb',
        color_type='color',
        imdecode_backend='cv2',
        key='gt',
        type='LoadImageFromFile'),
    dict(type='PackInputs'),
]
train_cfg = dict(
    max_iters=500000, type='IterBasedTrainLoop', val_interval=5000)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='meta_info_FFHQ6000sub_GT.txt',
        data_prefix=dict(gt='GT'),
        data_root='/home/featurize/data/ffhq/train',
        filename_tmpl=dict(gt='{}', img='{}'),
        pipeline=[
            dict(
                channel_order='rgb',
                color_type='color',
                imdecode_backend='cv2',
                key='gt',
                type='LoadImageFromFile'),
            dict(dst_keys=[
                'img',
            ], src_keys=[
                'gt',
            ], type='CopyValues'),
            dict(dictionary=dict(scale=4), type='SetValues'),
            dict(
                keys=[
                    'img',
                ],
                params=dict(
                    beta_gaussian=[
                        0.5,
                        4,
                    ],
                    beta_plateau=[
                        1,
                        2,
                    ],
                    kernel_list=[
                        'iso',
                        'aniso',
                        'generalized_iso',
                        'generalized_aniso',
                        'plateau_iso',
                        'plateau_aniso',
                    ],
                    kernel_prob=[
                        0.45,
                        0.25,
                        0.12,
                        0.03,
                        0.12,
                        0.03,
                    ],
                    kernel_range=[
                        7,
                        9,
                        11,
                        13,
                        15,
                        17,
                        19,
                        21,
                    ],
                    pad_to=21,
                    sigma_x=[
                        0.2,
                        3,
                    ],
                    sigma_y=[
                        0.2,
                        3,
                    ],
                    sinc_prob=0.1),
                type='RandomSecondOrderBlur'),
            dict(
                keys=[
                    'img',
                ],
                params=dict(
                    resize_mode_prob=[
                        0.2,
                        0.7,
                        0.1,
                    ],
                    resize_opt=[
                        'bicubic',
                        'bilinear',
                        'area',
                    ],
                    resize_prob=[
                        0.333,
                        0.333,
                        0.334,
                    ],
                    resize_scale=[
                        0.2,
                        1.5,
                    ]),
                type='RandomResize'),
            dict(
                keys=[
                    'img',
                ],
                params=dict(
                    gaussian_gray_noise_prob=0.4,
                    gaussian_sigma=[
                        1,
                        20,
                    ],
                    noise_prob=[
                        0.5,
                        0.5,
                    ],
                    noise_type=[
                        'gaussian',
                        'poisson',
                    ],
                    poisson_gray_noise_prob=0.4,
                    poisson_scale=[
                        0.05,
                        2,
                    ]),
                type='RandomNoise'),
            dict(
                keys=[
                    'img',
                ],
                params=dict(quality=[
                    50,
                    95,
                ]),
                type='RandomJPEGCompression'),
            dict(
                keys=[
                    'img',
                ],
                params=dict(
                    beta_gaussian=[
                        0.5,
                        4,
                    ],
                    beta_plateau=[
                        1,
                        2,
                    ],
                    kernel_list=[
                        'iso',
                        'aniso',
                        'generalized_iso',
                        'generalized_aniso',
                        'plateau_iso',
                        'plateau_aniso',
                    ],
                    kernel_prob=[
                        0.45,
                        0.25,
                        0.12,
                        0.03,
                        0.12,
                        0.03,
                    ],
                    kernel_range=[
                        7,
                        9,
                        11,
                        13,
                        15,
                        17,
                        19,
                        21,
                    ],
                    pad_to=21,
                    sigma_x=[
                        0.2,
                        1.5,
                    ],
                    sigma_y=[
                        0.2,
                        1.5,
                    ],
                    sinc_prob=0.1),
                type='RandomSecondOrderBlur'),
            dict(
                keys=[
                    'img',
                ],
                params=dict(
                    resize_mode_prob=[
                        0.3,
                        0.4,
                        0.3,
                    ],
                    resize_opt=[
                        'bicubic',
                        'bilinear',
                        'area',
                    ],
                    resize_prob=[
                        0.333,
                        0.333,
                        0.334,
                    ],
                    resize_scale=[
                        0.3,
                        1.2,
                    ]),
                type='RandomResize'),
            dict(
                keys=[
                    'img',
                ],
                params=dict(
                    gaussian_gray_noise_prob=0.4,
                    gaussian_sigma=[
                        1,
                        15,
                    ],
                    noise_prob=[
                        0.5,
                        0.5,
                    ],
                    noise_type=[
                        'gaussian',
                        'poisson',
                    ],
                    poisson_gray_noise_prob=0.4,
                    poisson_scale=[
                        0.05,
                        1.5,
                    ]),
                type='RandomNoise'),
            dict(
                keys=[
                    'img',
                ],
                params=dict(
                    final_since_prob=0.8,
                    order_prob=0.5,
                    quality=[
                        70,
                        95,
                    ],
                    target_size=(
                        128,
                        128,
                    )),
                type='FinalRandomSecondOrderDegradation'),
            dict(gt_patch_size=256, type='PairedRandomCrop'),
            dict(
                direction='horizontal',
                flip_ratio=0.5,
                keys=[
                    'img',
                    'gt',
                ],
                type='Flip'),
            dict(
                direction='vertical',
                flip_ratio=0.5,
                keys=[
                    'img',
                    'gt',
                ],
                type='Flip'),
            dict(
                keys=[
                    'img',
                    'gt',
                ],
                transpose_ratio=0.5,
                type='RandomTransposeHW'),
            dict(keys=[
                'img',
                'gt',
            ], type='PackInputs'),
        ],
        type='BasicImageDataset'),
    drop_last=True,
    num_workers=8,
    persistent_workers=False,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(
        channel_order='rgb',
        color_type='color',
        imdecode_backend='cv2',
        key='gt',
        type='LoadImageFromFile'),
    dict(dst_keys=[
        'img',
    ], src_keys=[
        'gt',
    ], type='CopyValues'),
    dict(dictionary=dict(scale=4), type='SetValues'),
    dict(
        keys=[
            'img',
        ],
        params=dict(
            beta_gaussian=[
                0.5,
                4,
            ],
            beta_plateau=[
                1,
                2,
            ],
            kernel_list=[
                'iso',
                'aniso',
                'generalized_iso',
                'generalized_aniso',
                'plateau_iso',
                'plateau_aniso',
            ],
            kernel_prob=[
                0.45,
                0.25,
                0.12,
                0.03,
                0.12,
                0.03,
            ],
            kernel_range=[
                7,
                9,
                11,
                13,
                15,
                17,
                19,
                21,
            ],
            pad_to=21,
            sigma_x=[
                0.2,
                3,
            ],
            sigma_y=[
                0.2,
                3,
            ],
            sinc_prob=0.1),
        type='RandomSecondOrderBlur'),
    dict(
        keys=[
            'img',
        ],
        params=dict(
            resize_mode_prob=[
                0.2,
                0.7,
                0.1,
            ],
            resize_opt=[
                'bicubic',
                'bilinear',
                'area',
            ],
            resize_prob=[
                0.333,
                0.333,
                0.334,
            ],
            resize_scale=[
                0.2,
                1.5,
            ]),
        type='RandomResize'),
    dict(
        keys=[
            'img',
        ],
        params=dict(
            gaussian_gray_noise_prob=0.4,
            gaussian_sigma=[
                1,
                20,
            ],
            noise_prob=[
                0.5,
                0.5,
            ],
            noise_type=[
                'gaussian',
                'poisson',
            ],
            poisson_gray_noise_prob=0.4,
            poisson_scale=[
                0.05,
                2,
            ]),
        type='RandomNoise'),
    dict(
        keys=[
            'img',
        ],
        params=dict(quality=[
            50,
            95,
        ]),
        type='RandomJPEGCompression'),
    dict(
        keys=[
            'img',
        ],
        params=dict(
            beta_gaussian=[
                0.5,
                4,
            ],
            beta_plateau=[
                1,
                2,
            ],
            kernel_list=[
                'iso',
                'aniso',
                'generalized_iso',
                'generalized_aniso',
                'plateau_iso',
                'plateau_aniso',
            ],
            kernel_prob=[
                0.45,
                0.25,
                0.12,
                0.03,
                0.12,
                0.03,
            ],
            kernel_range=[
                7,
                9,
                11,
                13,
                15,
                17,
                19,
                21,
            ],
            pad_to=21,
            sigma_x=[
                0.2,
                1.5,
            ],
            sigma_y=[
                0.2,
                1.5,
            ],
            sinc_prob=0.1),
        type='RandomSecondOrderBlur'),
    dict(
        keys=[
            'img',
        ],
        params=dict(
            resize_mode_prob=[
                0.3,
                0.4,
                0.3,
            ],
            resize_opt=[
                'bicubic',
                'bilinear',
                'area',
            ],
            resize_prob=[
                0.333,
                0.333,
                0.334,
            ],
            resize_scale=[
                0.3,
                1.2,
            ]),
        type='RandomResize'),
    dict(
        keys=[
            'img',
        ],
        params=dict(
            gaussian_gray_noise_prob=0.4,
            gaussian_sigma=[
                1,
                15,
            ],
            noise_prob=[
                0.5,
                0.5,
            ],
            noise_type=[
                'gaussian',
                'poisson',
            ],
            poisson_gray_noise_prob=0.4,
            poisson_scale=[
                0.05,
                1.5,
            ]),
        type='RandomNoise'),
    dict(
        keys=[
            'img',
        ],
        params=dict(
            final_since_prob=0.8,
            order_prob=0.5,
            quality=[
                70,
                95,
            ],
            target_size=(
                128,
                128,
            )),
        type='FinalRandomSecondOrderDegradation'),
    dict(gt_patch_size=256, type='PairedRandomCrop'),
    dict(
        direction='horizontal',
        flip_ratio=0.5,
        keys=[
            'img',
            'gt',
        ],
        type='Flip'),
    dict(
        direction='vertical',
        flip_ratio=0.5,
        keys=[
            'img',
            'gt',
        ],
        type='Flip'),
    dict(keys=[
        'img',
        'gt',
    ], transpose_ratio=0.5, type='RandomTransposeHW'),
    dict(keys=[
        'img',
        'gt',
    ], type='PackInputs'),
]
val_cfg = dict(type='MultiValLoop')
val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(gt='GT', img='LQ'),
        data_root='/home/featurize/data/ffhq/val',
        filename_tmpl=dict(gt='{}', img='{}'),
        pipeline=[
            dict(
                channel_order='rgb',
                color_type='color',
                imdecode_backend='cv2',
                key='img',
                type='LoadImageFromFile'),
            dict(
                channel_order='rgb',
                color_type='color',
                imdecode_backend='cv2',
                key='gt',
                type='LoadImageFromFile'),
            dict(keys=[
                'img',
                'gt',
            ], type='PackInputs'),
        ],
        type='BasicImageDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(type='MAE'),
    dict(crop_border=4, type='PSNR'),
    dict(crop_border=4, type='SSIM'),
]
val_pipeline = [
    dict(
        channel_order='rgb',
        color_type='color',
        imdecode_backend='cv2',
        key='img',
        type='LoadImageFromFile'),
    dict(
        channel_order='rgb',
        color_type='color',
        imdecode_backend='cv2',
        key='gt',
        type='LoadImageFromFile'),
    dict(keys=[
        'img',
        'gt',
    ], type='PackInputs'),
]
vis_backends = [
    dict(
        define_metric_cfg=[
            dict(name='lr', step_metric='iter'),
            dict(name='PSNR', step_metric='iter', step_sync=True),
            dict(name='MAE', step_metric='iter', step_sync=True),
            dict(name='SSIM', step_metric='iter', step_sync=True),
            dict(name='loss', step_metric='iter'),
        ],
        init_kwargs=dict(
            allow_val_change=True,
            name=
            'swinir_iter500000_psMULTISTEP_opAdam_daPRC_FlipH_FlipV_RT_nTrue',
            project='AI6126project2',
            resume='allow'),
        type='WandbVisBackend'),
]
visualizer = dict(
    bgr2rgb=True,
    fn_key='gt_path',
    img_keys=[
        'gt_img',
        'input',
        'pred_img',
    ],
    type='ConcatImageVisualizer',
    vis_backends=[
        dict(
            define_metric_cfg=[
                dict(name='lr', step_metric='iter'),
                dict(name='PSNR', step_metric='iter', step_sync=True),
                dict(name='MAE', step_metric='iter', step_sync=True),
                dict(name='SSIM', step_metric='iter', step_sync=True),
                dict(name='loss', step_metric='iter'),
            ],
            init_kwargs=dict(
                allow_val_change=True,
                name=
                'swinir_iter500000_psMULTISTEP_opAdam_daPRC_FlipH_FlipV_RT_nTrue',
                project='AI6126project2',
                resume='allow'),
            type='WandbVisBackend'),
    ])
work_dir = '/home/featurize/out/AI6126project2/swinir/swinir_iter500000_psMULTISTEP_opAdam_daPRC_FlipH_FlipV_RT_nTrue'
