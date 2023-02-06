import torch
from cleanfid import fid
from matplotlib import pyplot as plt
from torchvision.utils import save_image


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
    # Generate and save out latent space interpolations.
    with torch.no_grad():
        z = torch.randn(100, 128, dtype = torch.float).cuda()
        z[:,:2] -= torch.min(z[:,:2])
        z[:,:2] /= torch.max(z[:,:2])
        z[:,:2] = 2*z[:,:2]-1
        z[:,2:] = 0
        img = gen.forward_given_samples(z)
        save_image(img,path)
        
    return img
