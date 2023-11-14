import torch
import os

path = '/home/solashi/hungpv/compet/dacl/mmsegmentation/work_dirs/upernet_convnext_base_fp16_640x640_120k_all'
for name in os.listdir(path):
    if name.endswith('.pth'):
        state_dict = torch.load(f'{path}/{name}')['state_dict']
        meta = torch.load(f'{path}/{name}')['meta']
        torch.save({'state_dict': state_dict, 'meta': meta}, f'{path}/{name}')
