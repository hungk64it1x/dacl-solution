_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/dacl.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (640, 640)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth'  # noqa
main_loss = [
    dict(type='LovaszLoss', reduction='none', loss_weight=1.0),
]
# model settings
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 4, 18, 3]),

    decode_head=dict(in_channels=[64, 128, 320, 512], loss_decode=main_loss, num_classes=20),

    test_cfg=dict(mode='whole', crop_size=crop_size, stride=(426, 426))
)

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=4, workers_per_gpu=4)
work_dir = './work_dirs/segformer_mit-b3_fp16_640x640_80k_all_lovaszloss'