# Generative-Modeling
In this project three main generative modeling architectures are tested:

1. Basic GAN network architecture, the standard GAN [1](https://arxiv.org/pdf/1406.2661.pdf), LSGAN [2](https://arxiv.org/pdf/1611.04076.pdf) and WGAN-GP [3](https://arxiv.org/pdf/1704.00028.pdf)
2. Auto-encoder, variational auto-encoder (VAE) [4](https://arxiv.org/pdf/1606.05908.pdf) and a beta-VAE [5](https://arxiv.org/pdf/1804.03599.pdf) with a linear schedule.
3. DDPM [6](https://arxiv.org/abs/2006.11239) and DDIM [7](https://arxiv.org/abs/2010.02502) sampling for diffusion models.

## Dataset
The GAN model was trained and tested on CUB 2011 Dataset (http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) for faster training. This dataset contains 11,708 images of close-up shots of different bird species in various environments. Our models are trained to generate realistic-looking samples of these birds. Due to computational considerations for the course, we used a downsampled version of the dataset at a 32x32 resolution. The VAE and diffusion model was trained on CIFAR10 dataset for better utilizing avaliable pretrained model.

## GAN 
We implemented three loss functions: GAN loss from paper [1](https://arxiv.org/pdf/1406.2661.pdf), LSGAN loss from paper [2](https://arxiv.org/pdf/1611.04076.pdf), WGAN-GP loss from paper [3](https://arxiv.org/pdf/1704.00028.pdf). By comparing the result, WGAN-GP has the stablest training. It stabilizes training by adding continuous noise to the inputs of the discriminator, and transfer the gradient of the objective function to JS-Divergence which is beneficial in GAN training, because even when discriminator is at its optimal the generator’s gradient will not vanish. Secondly, they also utilized the Wasserstein distance which has a smoother gradient everywhere, and it learns no matter the performance of the generator. Lastly, they introduced gradient penalty to enforce Lipschitz constraint. Compared to gradient penalty, weight clipping are very sensitive to the change of hyper parameter c which can cause either vanishing gradient or exploding gradient problem. Furthermore, since weight clipping acting as a weight regularization, it limits the model’s capacity of learning complex function. 

## VAE
Traditional autoencoder, VAE and Beta VAE were trained and tested. By tunning beta, and beta value scheduler we were able to produce the best results compared to the vanilla VAE.

## Diffusion
In this project we trained and tested Denoising Diffusion Probabilistic Models (DDPM), and Denoising Diffusion Implicit Model. For DDPM, increase number of sampling steps improve the overall image quality but requires longer to sample and generate results. So that the issue with DDPM is that we need to loop over all the timestamps sequentially, which is not very efficient. Denoising Diffusion Implicit Model (DDIM) samples a small number of timesteps from the total timestamps. But it reauires hyperparameter tuning to yield comparable results with DDPM.


## Paper Cited
1. Generative Adversarial Nets (Goodfellow et al, 2014): https://arxiv.org/pdf/1406.2661.pdf

2. Least Squares Generative Adversarial Networks (Mao et al, 2016): https://arxiv.org/pdf/1611.04076.pdf

3. Improved Training of Wasserstein GANs (Gulrajani et al, 2017): https://arxiv.org/pdf/1704.00028.pdf

4. Tutorial on Variational Autoencoders (Doersch, 2016): https://arxiv.org/pdf/1606.05908.pdf

5. Understanding disentangling in β-VAE (Burgess et al, 2018): https://arxiv.org/pdf/1804.03599.pdf

6. Denoising diffusion probabilistic models (Jonathan Ho, et al, 2020): https://arxiv.org/abs/2006.11239

7. Denoising diffusion implicit models (Jiaming Song et al, 2020): https://arxiv.org/abs/2010.02502

8. What are diffusion models? Lil’Log. (Lilian Weng, 2021): https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
