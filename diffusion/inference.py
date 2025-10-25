import argparse
import os
import torch

from model import DiffusionModel
from unet import Unet
from torchvision.utils import save_image
from cleanfid import fid as cleanfid

@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    
    ##################################################################
    # TODO 3.3: Write a function that samples images from the
    # diffusion model given z
    # Note: The output must be in the range [0, 255]!
    ##################################################################
    def gen_fn(z):
        print(f'inside gen_fn: {z.shape}')
        
        # z_shape = (curr_batch_size, 3, dataset_resolution, dataset_resolution)
        # gen_output = gen.sample_given_z(z, z_shape)
        
    # gen_fn = None
    # all_samples = list()
    # import numpy as np
    # total = 0
    # while total < num_gen:
    #     curr_batch_size = min(batch_size, num_gen - len(all_samples))
    #     z_shape = (curr_batch_size, 3, dataset_resolution, dataset_resolution)
    #     device = next(gen.parameters()).device

    #     z = torch.randn(curr_batch_size*z_dimension, device=device)
    #     gen_fn = gen.sample_given_z(z, z_shape)

    #     gen_fn = torch.clamp(gen_fn, 0, 1) * 255
    #     gen_fn = gen_fn.byte().permute(0, 2, 3, 1).cpu().numpy()
    #     all_samples.append(gen_fn)

    #     total += curr_batch_size
    #     print(f'processed {total} imgs out of {num_gen}')

    # gen_fn = np.concatenate(all_samples, axis=0)
        # convert to numpy and get in b,h,w,c format
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    score = cleanfid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="train",
    )
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diffusion Model Inference')
    parser.add_argument('--ckpt', required=True, type=str, help="Pretrained checkpoint")
    parser.add_argument('--num-images', default=100, type=int, help="Number of images per iteration")
    parser.add_argument('--image-size', default=32, type=int, help="Image size to generate")
    parser.add_argument('--sampling-method', choices=['ddpm', 'ddim'])
    parser.add_argument('--ddim-timesteps', type=int, default=25, help="Number of timesteps to sample for DDIM")
    parser.add_argument('--ddim-eta', type=int, default=1, help="Eta for DDIM")
    parser.add_argument('--compute-fid', action="store_true")
    args = parser.parse_args()

    prefix = f"data_{args.sampling_method}/"
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    sampling_timesteps = args.ddim_timesteps if args.sampling_method == "ddim" else None

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    ).cuda()
    diffusion = DiffusionModel(
        model,
        timesteps=1000,   # number of timesteps
        sampling_timesteps=sampling_timesteps,
        ddim_sampling_eta=args.ddim_eta,
    ).cuda()

    img_shape = (args.num_images, diffusion.channels, args.image_size, args.image_size)

    # load pre-trained weight
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt["model_state_dict"])

    with torch.no_grad():
        # run inference
        model.eval()
        if args.sampling_method == "ddpm":
            generated_samples = diffusion.sample(img_shape)
        elif args.sampling_method == "ddim":
            generated_samples = diffusion.sample(img_shape)
        save_image(
            generated_samples.data.float(),
            prefix + f"samples_{args.sampling_method}.png",
            nrow=10,
        )
        if args.compute_fid:
            # NOTE: This will take a very long time to run even though we are only doing 10K samples.
            score = get_fid(diffusion, "cifar10", 32, 32*32*3, batch_size=256, num_gen=10_000)
            print("FID: ", score)
