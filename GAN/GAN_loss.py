import os

import torch

from networks import Discriminator, Generator
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    # TODO 1.3.1: Implement GAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders for Q1.5.
    criterion = nn.BCEWithLogitsLoss()
    label_real = Variable(torch.cuda.FloatTensor(discrim_real.size(0),1).fill_(1.0), requires_grad=False)
    real_loss = criterion(discrim_real,label_real)
    label_fake = Variable(torch.cuda.FloatTensor(discrim_fake.size(0),1).fill_(0.0), requires_grad=False)
    fake_loss = criterion(discrim_fake,label_fake)
    loss = (fake_loss + real_loss)/2
    return loss


def compute_generator_loss(discrim_fake):
    # TODO 1.3.1: Implement GAN loss for generator.
    criterion = nn.BCEWithLogitsLoss()
    label = Variable(torch.cuda.FloatTensor(discrim_fake.size(0),1).fill_(1.0), requires_grad=False)
    loss = criterion(discrim_fake,label)

    return loss


if __name__ == "__main__":

    os.environ["PYTORCH_JIT"] = "0"
    gen = Generator().cuda().to(memory_format=torch.channels_last)
    disc = Discriminator().cuda().to(memory_format=torch.channels_last)

    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.3.2: Run this line of code.
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