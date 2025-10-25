import argparse
import torch
from cleanfid import fid
from matplotlib import pyplot as plt


def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename + ".png")


@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    gen_fn = lambda z: (gen.forward_given_samples(z) / 2 + 0.5) * 255
    score = fid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="custom",
    )
    return score


@torch.no_grad()
def interpolate_latent_space(gen, path):
    ##################################################################
    # TODO: 1.2: Generate and save out latent space interpolations.
    # 1. Generate 100 samples of 128-dim vectors. Do so by linearly
    # interpolating for 10 steps across each of the first two
    # dimensions between -1 and 1. Keep the rest of the z vector for
    # the samples to be some fixed value (e.g. 0).
    # 2. Forward the samples through the generator.
    # 3. Save out an image holding all 100 samples.
    # Use torchvision.utils.save_image to save out the visualization.
    ##################################################################
    # ok so we need 100 128 dim latent vectors: shape is 100,128
    # Linearly interpolate from -1 to 1
    lin = torch.linspace(-1, 1, 10)
    
    # Create a batch of 100 z vectors
    row=0
    Z = torch.zeros(100, 128, device=next(gen.parameters()).device)
    for i in range(10):
        for j in range(10):
            Z[row,0] = lin[i]
            Z[row, 1] = lin[j]
            row += 1
    gen_output = gen.forward_given_samples(Z)
    # 100,3,32,32 output?
    gen_output = (gen_output + 1) / 2  # from [-1,1] → [0,1]

    # Save image grid (10x10)
    from torchvision.utils import save_image
    save_image(gen_output, path, nrow=10)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_amp", action="store_true")
    args = parser.parse_args()
    return args
