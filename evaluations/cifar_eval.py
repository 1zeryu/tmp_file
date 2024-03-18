import argparse
import io
import os
import pdb
import random
import warnings
import zipfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Iterable, Optional, Tuple
from .evaluator import Evaluator, FIDStatistics
import numpy as np
import requests
import tensorflow.compat.v1 as tf
from scipy import linalg
from tqdm.auto import tqdm



INCEPTION_V3_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb"
INCEPTION_V3_PATH = "classify_image_graph_def.pb"

FID_POOL_NAME = "pool_3:0"
FID_SPATIAL_NAME = "mixed_6/conv:0"


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("ref_batch", help="path to reference batch npz file", )
    parser.add_argument("sample_batch", help="path to sample batch npz file")
    parser.add_argument('--refer_batch', default='/home/wangkai/diffusion_acceleration/improved-diffusion/datasets/ref/fid_stats_cifar10_train.npz')
    args = parser.parse_args()

    config = tf.ConfigProto(
        allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
    )
    config.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config=config))

    print("warming up TensorFlow...")
    # This will cause TF to print a bunch of verbose stuff now rather
    # than after the next print(), to help prevent confusion.
    evaluator.warmup()

    print("computing sample batch activations...")
    sample_acts = evaluator.read_activations(args.sample_batch)
    print("computing/reading sample batch statistics...")
    sample_stats, sample_stats_spatial = evaluator.read_statistics(args.sample_batch, sample_acts)
    pdb.set_trace()

    print("computing reference batch activations...")
    state = np.load(args.ref_batch)
    mu, sigma = state["mu"], state["sigma"]
    ref_stats = FIDStatistics(mu, sigma)

    print("Computing evaluations...")
    print("Inception Score:", evaluator.compute_inception_score(sample_acts[0]))
    print("FID:", sample_stats.frechet_distance(ref_stats))