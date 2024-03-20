import numpy as np
from PIL import Image
import os

def uncomp_npz(npz_path, img_dir):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    with np.load(npz_path) as data:
         data = data['arr_0']
         for i, arr in enumerate(data):
             # save in directory
            img = Image.fromarray(arr)
            img.save(os.path.join(img_dir, f"{i}.png"))


    print("done")


npz_path = 'ref/VIRTUAL_imagenet256_labeled.npz'
img_dir = 'ref/VIRTUAL_imagenet256_labeled'

uncomp_npz(npz_path, img_dir)