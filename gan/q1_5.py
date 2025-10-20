import os

import torch
from utils import get_args

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    """
    TODO 1.5.1: Implement WGAN-GP loss for discriminator.
    loss = E[D(fake_data)] - E[D(real_data)] + lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]
    """
    ##################################################################
    # TODO 1.5: Implement WGAN-GP loss for discriminator.
    # loss_pt1 = E[D(fake_data)] - E[D(real_data)]
    # loss_pt2 = lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]
    # loss = loss_pt1 + loss_pt2
    ##################################################################
    loss = None
    loss_pt1 = torch.mean(discrim_fake) - torch.mean(discrim_real)

    # take gradient of D(x_interp) wrt x_interp
    # output should be same shape as x_interp, b,c,h,w
    gradients = torch.autograd.grad(
        outputs=discrim_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(discrim_interp),
        create_graph=True
    )[0]
    # output should be same shape b,c,h,w
    # reshape so we have per sample b,c*h*w
    gradients = gradients.view(gradients.shape[0], -1)
    # take l2 norm
    norm = torch.norm(gradients, dim=1)
    loss_pt2 = (lamb * (norm - 1)**2).mean()  # take mean across batch dim
    loss = loss_pt1 + loss_pt2
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO 1.5: Implement WGAN-GP loss for generator.
    # loss = - E[D(fake_data)]
    ##################################################################
    loss = None
    loss = -torch.mean(discrim_fake)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_wgan_gp/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
