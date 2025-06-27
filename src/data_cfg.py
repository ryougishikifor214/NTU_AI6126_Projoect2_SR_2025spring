from project_config import DATASET_TRAIN_DIR_PATH, DATASET_VAL_DIR_PATH

scale = 4
gt_h_size = 512
gt_w_size = 512

train_pipeline = [
    dict(type='LoadImageFromFile', 
         key='gt',
         color_type='color', 
         channel_order='rgb',
         imdecode_backend='cv2'),
    dict(type= 'CopyValues',
         src_keys=['gt'], 
         dst_keys=['img']),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    
    ## first-order degradadtion
    dict(type='RandomSecondOrderBlur', 
         keys=['img'],
         params = dict(
               kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
               kernel_prob=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
               sinc_prob = 0.1,
               pad_to = 21,
               kernel_range = [2 * v + 1 for v in range(3, 11)],
               sigma_x = [0.2, 3],
               sigma_y = [0.2, 3],
               beta_gaussian = [0.5, 4],
               beta_plateau = [1, 2],
         ),
     ),
    dict(type='RandomResize', 
         keys=['img'], 
         params = dict(
             resize_opt = ['bicubic', 'bilinear', 'area'],
             resize_prob = [0.333, 0.333, 0.334],
             resize_mode_prob = [0.2, 0.7, 0.1], #up, down, keep
             resize_scale = [0.2, 1.5],
         ),
    ),
    dict(type='RandomNoise', 
         keys=['img'], 
         params=dict(
             noise_type = ['gaussian', 'poisson'],
             noise_prob = [0.5, 0.5],
             gaussian_sigma = [1, 20],
             gaussian_gray_noise_prob = 0.4,
             poisson_scale = [0.05, 2],
             poisson_gray_noise_prob = 0.4
         ),
    ),
    dict(type='RandomJPEGCompression', 
         keys=['img'], 
         params = dict(
             quality = [50, 95],
         ),
    ),
    
    ## second-order degradation
    dict(type='RandomSecondOrderBlur', 
         keys=['img'],
         params = dict(
               kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
               kernel_prob=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
               sinc_prob = 0.1,
               pad_to = 21,
               kernel_range = [2 * v + 1 for v in range(3, 11)],
               sigma_x = [0.2, 1.5],
               sigma_y = [0.2, 1.5],
               beta_gaussian = [0.5, 4],
               beta_plateau = [1, 2],
         ),
     ),
    dict(type='RandomResize', 
         keys=['img'], 
         params = dict(
             resize_opt = ['bicubic', 'bilinear', 'area'],
             resize_prob = [0.333, 0.333, 0.334],
             resize_mode_prob = [0.3, 0.4, 0.3], #up, down, keep
             resize_scale = [0.3, 1.2],
        ),
    ),
    dict(type='RandomNoise', 
         keys=['img'], 
         params=dict(
             noise_type = ['gaussian', 'poisson'],
             noise_prob = [0.5, 0.5],
             gaussian_sigma = [1, 15],
             gaussian_gray_noise_prob = 0.4,
             poisson_scale = [0.05, 1.5],
             poisson_gray_noise_prob = 0.4
         ),
    ),
    dict(type='FinalRandomSecondOrderDegradation', 
         keys=['img'], 
         params = dict(
             order_prob = 0.5,
             quality = [70, 95],
             target_size = (gt_h_size//4, gt_w_size//4),
             final_since_prob = 0.8,
         ),
    ),
    
    # dict(type='PairedRandomCrop', gt_patch_size=196), #edsr
    # dict(
    #     type='Flip',
    #     keys=['img', 'gt'],
    #     flip_ratio=0.5,
    #     direction='horizontal'),
    # # dict(
    # #     type='RandomRotationWithRatio',
    # #     keys=['img', 'gt'],
    # #     degrees=15,
    # #     rotate_ratio=0.3
    # #  ),
    # dict(
    #     type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    # dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    # dict(type='Cutblur', 
    #      params=dict(
    #          alpha=0.7),),
    dict(type='PackInputs', keys=['img', 'gt']),
]

val_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(type='PackInputs'),
]

dataset_type = 'BasicImageDataset'
train_dataloader = dict(
     num_workers=8,
     batch_size=16,
     drop_last=True,
     persistent_workers=False,
     sampler=dict(type='InfiniteSampler', shuffle=True),
     dataset = dict(
          type=dataset_type,
          ann_file='meta_info_FFHQ6000sub_GT.txt',
          data_root = DATASET_TRAIN_DIR_PATH,
          data_prefix = dict(gt='GT'),
          filename_tmpl=dict(img='{}', gt='{}'),
          pipeline=train_pipeline,
     ),
     _delete_=True,
)

val_dataloader = dict(
     num_workers=4,
     persistent_workers=False,
     drop_last=False,
     sampler=dict(type='DefaultSampler', shuffle=False),
     dataset = dict(
          type=dataset_type,
          data_root = DATASET_VAL_DIR_PATH,
          data_prefix = dict(img='LQ', gt='GT'),
          filename_tmpl=dict(img='{}', gt='{}'),
          pipeline=val_pipeline,
     ),
     _delete_=True,
)

val_evaluator = [
    dict(type='MAE'),  # The name of metrics to evaluate
    dict(type='PSNR', crop_border=scale),  # The name of metrics to evaluate
    dict(type='SSIM', crop_border=scale),  # The name of metrics to evaluate
]

test_dataloader = val_dataloader
test_evaluator = val_evaluator

