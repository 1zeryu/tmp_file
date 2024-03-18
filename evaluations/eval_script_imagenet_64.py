BASE="/home/wangkai/diffusion_acceleration/improved-diffusion/outputs/imagenet64_base"
TRUNC="/home/wangkai/diffusion_acceleration/improved-diffusion/outputs/imagenet64_trunc"
COSINE="/home/wangkai/diffusion_acceleration/improved-diffusion/outputs/imagenet64_cosine"
SPEEDUP="/home/wangkai/diffusion_acceleration/guided-diffusion/outputs/imagenet64_speedup"

REF="/home/wangkai/diffusion_acceleration/improved-diffusion/datasets/ref/VIRTUAL_imagenet64_labeled.npz"
# test BASE

class_name = ['base_', 'trunc_', 'cosine_', 'speedup_']

import os
from glob import glob

print("test base")
def eval(base_path, cls):
    npz_file = glob(os.path.join(base_path, "*.npz"))
    for npz_name in npz_file:
        npz_name = npz_name.split("/")[-1]
        print(npz_name)
        os.system(f"python evaluator.py {REF} {os.path.join(base_path, npz_name)} > {class_name[cls] + npz_name + '.txt'}")

eval(BASE, 0)
eval(TRUNC, 1)
eval(COSINE, 2)
eval(SPEEDUP, 3)

