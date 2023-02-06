from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from model import AEModel
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import time
import os
from utils import *


def ae_loss(model, x):

    latent = model.encoder(x)
    recon = model.decoder(latent)
    loss = F.mse_loss(recon,x,reduction = 'none').view(x.shape[0], -1).sum(1).mean()

    return loss, OrderedDict(recon_loss=loss)

def vae_loss(model, x, beta = 1):

    mu,log_std = model.encoder(x)
    std = torch.exp(0.5*log_std)
    dist = torch.distributions.Normal(mu,std)
    latent = dist.rsample()
    recon = model.decoder(latent)
    recon_loss = F.mse_loss(recon,x,reduction = 'none').view(x.shape[0], -1).sum(1).mean()
    kl_loss = torch.mean(-0.5 * torch.sum(1 + log_std - mu ** 2 - log_std.exp(), dim = 1), dim = 0)

    total_loss = recon_loss + beta*kl_loss
    return total_loss, OrderedDict(recon_loss=recon_loss, kl_loss=kl_loss)


def constant_beta_scheduler(target_val = 1):
    def _helper(epoch):
        return target_val
    return _helper

def linear_beta_scheduler(max_epochs=None, target_val = 1):

    def _helper(epoch):
        beta = epoch/(max_epochs-1) * target_val
        return beta
    return _helper

def run_train_epoch(model, loss_mode, train_loader, optimizer, beta = 1, grad_clip = 1):
    model.train()
    all_metrics = []
    for x, _ in train_loader:
        x = preprocess_data(x)
        if loss_mode == 'ae':
            loss, _metric = ae_loss(model, x)
        elif loss_mode == 'vae':
            loss, _metric = vae_loss(model, x, beta)
        all_metrics.append(_metric)
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return avg_dict(all_metrics)


def get_val_metrics(model, loss_mode, val_loader):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = preprocess_data(x)
            if loss_mode == 'ae':
                _, _metric = ae_loss(model, x)
            elif loss_mode == 'vae':
                _, _metric = vae_loss(model, x)
            all_metrics.append(_metric)

    return avg_dict(all_metrics)

def main(log_dir, loss_mode = 'vae', beta_mode = 'constant', num_epochs = 20, batch_size = 256, latent_size = 256,
         target_beta_val = 1, grad_clip=1, lr = 1e-3, eval_interval = 5):

    os.makedirs('data/'+ log_dir, exist_ok = True)
    train_loader, val_loader = get_dataloaders()

    variational = True if loss_mode == 'vae' else False
    model = AEModel(variational, latent_size, input_shape = (3, 32, 32)).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    vis_x = next(iter(val_loader))[0][:36]
    val_loss = []
    val_kldloss = []
    
    if beta_mode == 'constant':
        beta_fn = constant_beta_scheduler(target_val = target_beta_val) 
    elif beta_mode == 'linear':
        beta_fn = linear_beta_scheduler(max_epochs=num_epochs, target_val = target_beta_val) 

    for epoch in range(num_epochs):
        print('epoch', epoch)
        train_metrics = run_train_epoch(model, loss_mode, train_loader, optimizer, beta_fn(epoch))
        val_metrics = get_val_metrics(model, loss_mode, val_loader)
        val_loss.append(val_metrics['recon_loss'])
        if loss_mode == 'vae':
            val_kldloss.append(val_metrics['kl_loss'])

        if (epoch+1)%eval_interval == 0:
            print(epoch, train_metrics)
            print(epoch, val_metrics)

            vis_recons(model, vis_x, 'data/'+log_dir+ '/epoch_'+str(epoch))
            if loss_mode == 'vae':
                vis_samples(model, 'data/'+log_dir+ '/epoch_'+str(epoch) )
    return val_loss,val_kldloss




if __name__ == '__main__':
    #Auto-Encoder
    #Run for latent_sizes 16, 128 and 1024
    # latent_size = [16,128,1024]
    # val_loss = []
    # for size in latent_size:
    #     log_dir = 'ae_latent' + str(size)
    #     val_loss.append(main(log_dir, loss_mode = 'ae',  num_epochs = 20, latent_size = size)[0])
    # fig = plt.figure()
    # for i in range (len(val_loss)):
    #     plt.plot(np.arange(20),val_loss[i])
    #     plt.legend(['16','128','1024'])
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Recon_Loss')
    #     plt.title('Validation Loss vs Epoch')
    # plt.savefig('Validation Loss vs Epoch.png')

    # - Variational Auto-Encoder
    # val_loss,val_kldloss = main('vae_latent1024', loss_mode = 'vae', num_epochs = 20, latent_size = 900)
    # fig = plt.figure()
    # plt.plot(np.arange(20),val_loss)
    # plt.xlabel('Epoch')
    # plt.ylabel('Recon_Loss')
    # plt.title('Validation Loss vs Epoch')
    # plt.savefig('Validation Loss vs Epoch(VAE_Recon).png')
    # fig = plt.figure()
    # plt.plot(np.arange(20),val_kldloss)
    # plt.xlabel('Epoch')
    # plt.ylabel('KL Divergence')
    # plt.title('Validation Loss vs Epoch')
    # plt.savefig('Validation Loss vs Epoch(VAE_KLD).png')


    # - Beta-VAE (constant beta)
    #Run for beta values 0.8, 1.2
    # val_loss,val_kldloss = main('vae_latent1024_beta_constant1.2', loss_mode = 'vae', beta_mode = 'constant', target_beta_val = 1.2, num_epochs = 20, latent_size = 800)
    # fig = plt.figure()
    # plt.plot(np.arange(20),val_loss)
    # plt.xlabel('Epoch')
    # plt.ylabel('Recon_Loss')
    # plt.title('Validation Loss vs Epoch')
    # plt.savefig('Validation Loss vs Epoch(Beta1.2).png')
    # fig = plt.figure()
    # plt.plot(np.arange(20),val_kldloss)
    # plt.xlabel('Epoch')
    # plt.ylabel('KL Divergence')
    # plt.title('KL Divergence vs Epoch')
    # plt.savefig('KL Divergence vs Epoch(Beta1.2).png')

    # - VAE with annealed beta (linear schedule)
    val_loss,val_kldloss = main(
        'vae_latent1024_beta_linear1', loss_mode = 'vae', beta_mode = 'linear', 
        target_beta_val = 0.7, num_epochs = 20, latent_size = 1024
    )
    fig = plt.figure()
    plt.plot(np.arange(20),val_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Recon_Loss')
    plt.title('Validation Loss vs Epoch')
    plt.savefig('Validation Loss vs Epoch(Annealing Beta).png')
    fig = plt.figure()
    plt.plot(np.arange(20),val_kldloss)
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence vs Epoch')
    plt.savefig('KL Divergence vs Epoch(Annealing Beta).png')
