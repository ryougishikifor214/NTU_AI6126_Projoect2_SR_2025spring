from project_config import DATASET_TRAIN_DIR_PATH, DATASET_VAL_DIR_PATH

scale = 4
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
    
    
    dict(type='PackInputs')
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
    dict(type='PackInputs')
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
     )
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
     )
)

val_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='MAE'),
        dict(type='PSNR', crop_border=scale),
        dict(type='SSIM', crop_border=scale),
    ]
)

