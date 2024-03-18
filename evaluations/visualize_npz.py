import pdb
from PIL import Image
from evaluator import *
import matplotlib.pyplot as plt

visualze_dir = "/home/kwang/zhouyukun/diffusion_accelerator/guided-diffusion/outputs/visualize"
visualze_dir = "/home/wangkai/diffusion_acceleration/guided-diffusion/outputs/visualize"

import os

if not os.path.exists(visualze_dir):
    os.makedirs(visualze_dir)

def visualize(npz_file, name):
    print(npz_file)
    with open_npz_array(npz_file, "arr_0") as reader:
        for images in reader.read_batches(9):
            # pdb.set_trace()
            # transfer to n x n grid of images
            # pdb.set_trace()

            # transfer to 3 x 3 grid of images
            plt.figure(figsize=(3, 3))
            for i, img in enumerate(images):
                plt.subplot(3, 3, i+1)
                plt.imshow(img)
                plt.axis("off")
            plt.savefig(os.path.join(visualze_dir, name + ".png"))
            break

    print("images saved at", os.path.join(visualze_dir, name + ".png"))
    return



# imagenet64 base: /home/wangkai/diffusion_acceleration/improved-diffusion/outputs/imagenet64_base/
# imagenet64 trunc: /home/wangkai/diffusion_acceleration/improved-diffusion/outputs/imagenet64_trunc/
# imagenet64 cosine: /home/wangkai/diffusion_acceleration/improved-diffusion/outputs/imagenet64_cosine/
# imagenet64 speedup: /home/wangkai/diffusion_acceleration/guided-diffusion/outputs/imagenet64_speedup/

from glob import glob
def os_npz_file(npz_dir):
    npz_file = glob(os.path.join(npz_dir, "*.npz"))
    return npz_file

def main(npz_dir):
    npz_file = os_npz_file(npz_dir)
    for npz in npz_file:
        name = npz_dir.split("/")[-1] + "_" + npz.split("/")[-1].split(".")[1]
        visualize(npz, name)



if __name__ == "__main__":
    # main("/home/kwang/zhouyukun/diffusion_accelerator/guided-diffusion/outputs/imagenet_base")
    # main('/home/kwang/zhouyukun/diffusion_accelerator/guided-diffusion/outputs/imagenet_speedup')
    # main("/home/wangkai/diffusion_acceleration/improved-diffusion/outputs/imagenet64_base")
    main("/home/wangkai/diffusion_acceleration/improved-diffusion/outputs/imagenet64_trunc")
    # main("/home/wangkai/diffusion_acceleration/improved-diffusion/outputs/imagenet64_cosine")
    # main("/home/wangkai/diffusion_acceleration/guided-diffusion/outputs/imagenet64_speedup")