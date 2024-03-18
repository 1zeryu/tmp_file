import os

def inference(args, ckpt_dir, image_size, num_fid_samples, nproc_per_node, sample_dir, model='DiT-XL/2', faster=False):
    # list ckpt dir
    ckpt_list = os.listdir(ckpt_dir)
    # ckpt_list.remove('samples')
    print(ckpt_list)
    ckpt_list = sorted(ckpt_list, key=lambda x: x.split('-')[-1].split('.')[0])
    import pdb;pdb.set_trace()
    ckpt_list = [os.path.join(ckpt_dir, ckpt) for ckpt in ckpt_list if ckpt.endswith('.pt')]

    # other args
    master_port = 30000

    # generate samples
    for ckpt in ckpt_list:
        # run sample script
        print("running ckpt {}".format(ckpt))
        if not faster:
            cmd = f"torchrun --nnodes=1 --nproc_per_node={nproc_per_node} --master_port {master_port} sample_ddp.py --model {model} " \
                  f"--num-fid-samples {num_fid_samples} --ckpt {ckpt} --image-size {image_size} --sample-dir {sample_dir} --cfg-scale {args.cfg_scale}"
        else:
            cmd = f"torchrun --nnodes=1 --nproc_per_node={nproc_per_node} --master_port {master_port} sample_ddp.py --model {model} --num-fid-samples {num_fid_samples} --ckpt {ckpt} " \
                  f"--image-size {image_size} --sample-dir {sample_dir} --cfg-scale {args.cfg_scale}"

        os.system(cmd)
        print("Sample script finished")

def fid(ckpt_dir, fid_script_path, fid_ref, model='DiT-XL/2', image_size=256, cfg_scale=1):
    # list ckpt dir
    ckpt_list = os.listdir(ckpt_dir)
    ckpt_list = sorted(ckpt_list, key=lambda x: x.split('-')[-1].split('.')[0])
    import pdb; pdb.set_trace()
    ckpt_list = [os.path.join(ckpt_dir, ckpt) for ckpt in ckpt_list if ckpt.endswith('.pt')]


    # generate samples
    for ckpt in ckpt_list:
        ckpt_step = os.path.basename(ckpt).split('.')[0]
        model = model.replace('/', '-')
        sample_path = f'{model}-{ckpt_step}-size-256-vae-ema-cfg-{cfg_scale}-seed-0'
        sample_path = os.path.join('samples', sample_path)
        sample_npz = sample_path + '.npz'
        log = sample_path + '.log'
        # run fid script
        fid_cmd = f"python {fid_script_path} {fid_ref} {sample_npz} > {log}.log"
        os.system(fid_cmd)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', type=str, required=True)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--num-fid-samples', type=int, default=50000)
    parser.add_argument('--nproc-per-node', type=int, default=8)
    parser.add_argument('--sample-dir', type=str, default='samples')
    parser.add_argument('--model', type=str, default='DiT-XL/2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cfg-scale', type=float, default=1)
    parser.add_argument('--faster', action='store_true', default=False)
    args = parser.parse_args()

    if args.faster:
        args.sample_dir = os.path.join(args.sample_dir, 'faster')
    else:
        args.sample_dir = os.path.join(args.sample_dir, 'base')

    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    inference(args, args.ckpt_dir, args.image_size, args.num_fid_samples, args.nproc_per_node, args.sample_dir, args.model, args.faster)
    fid(args.ckpt_dir, 'evaluations/evaluator.py', 'evaluations/ref/VIRTUAL_imagenet256_labeled.npz', args.model, args.image_size, args.cfg_scale)

