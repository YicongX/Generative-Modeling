import os

import torch

from networks import Discriminator, Generator
from torch.autograd import Variable
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    # Implement WGAN-GP loss for discriminator.
    # loss = E[D(fake_data)] - E[D(real_data)] + lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]
    fake = Variable(torch.cuda.FloatTensor(interp.shape[0], 1).fill_(1.0), requires_grad=False)
    gradients = torch.autograd.grad(outputs=discrim_interp, 
                                        inputs=interp,
                                        grad_outputs=fake,
                                        create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(gradients.shape[0], -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    loss = -torch.mean(discrim_real) + torch.mean(discrim_fake) + lamb * gp
    return loss


def compute_generator_loss(discrim_fake):
    # Implement WGAN-GP loss for generator.
    # loss = - E[D(fake_data)]
    loss = -torch.mean(discrim_fake)
    return loss


if __name__ == "__main__":
    gen = Generator().cuda().to(memory_format=torch.channels_last)
    disc = Discriminator().cuda().to(memory_format=torch.channels_last)
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
    )
