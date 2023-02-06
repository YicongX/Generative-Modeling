import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    cosine_beta_schedule,
    default,
    extract,
    unnormalize_to_zero_to_one,
)

class DiffusionModel(nn.Module):
    def __init__(
        self,
        model,
        timesteps = 1000,
        sampling_timesteps = None,
        ddim_sampling_eta = 1.
    ):
        super(DiffusionModel, self).__init__()
        assert model.channels == model.out_dim
        assert not model.learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.device = torch.cuda.current_device()

        self.betas = cosine_beta_schedule(timesteps).to(self.device)
        self.num_timesteps = self.betas.shape[0]

        alphas = 1. - self.betas
        # compute the cummulative products for current and previous timesteps
        
        self.alphas_cumprod = torch.cumprod(alphas, axis = 0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value = 1.)
        

        # pre-compute the alphas needed for forward process
        self.alphas_cumprod_sqrt = torch.sqrt(self.alphas_cumprod)
        self.oneminus_alphas_cumprod_sqrt = torch.sqrt(1. - self.alphas_cumprod)
        self.recip_alphas_cumprod_sqrt = torch.sqrt(1. / self.alphas_cumprod)
        # calculations for posterior q(x_{t-1} | x_t, x_0) in DDPM
        self.posterior_variance = self._get_posterior_variance()
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min =1e-20))

        # compute the coefficients for the mean
        # This is coefficient of x_0 in the DDPM section
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        # This is coefficient of x_t in the DDPM section
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - self.alphas_cumprod)

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
    
    def _get_posterior_variance(self):
        # compute the variance of the posterior distribution
        pos_var = self.betas * (1. - self.alphas_cumprod_prev)/(1. - self.alphas_cumprod)
        return pos_var

    def predict_start_image_from_noise(self, x_t, t, noise):
        # given a noised image x_t and noise tensor, predict x_0
        x_start = extract(self.recip_alphas_cumprod_sqrt,t,x_t.shape) * x_t - extract(self.recip_alphas_cumprod_sqrt*self.oneminus_alphas_cumprod_sqrt,t,x_t.shape) * noise
        
        return x_start

    def get_posterior_parameters(self, x_start, x_t, t):
        # Compute the posterior mean and variance for x_{t-1} 
        # using the coefficients, x_t, and x_0
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t):
        # given a noised image x_t, predict x_0 and the additive noise
        pred_noise = self.model(x,t)
        x_start = self.predict_start_image_from_noise(x, t, pred_noise)

        return (pred_noise, x_start)

    def mean_variance_at_previous_timestep(self, x, t):
        # predict the mean and variance for the posterior (x_{t-1})
        x_start = self.model_predictions(x,t)[1]
        x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.get_posterior_parameters(x_start,x,t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def predict_denoised_at_prev_timestep(self, x, t: int):
        # given x at timestep t, predict the denoised image at x_{t-1}.
        # also return the predicted starting image.
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        mean, _,log_var, x_start = self.mean_variance_at_previous_timestep(x,t)
        pred_img = mean + torch.exp(0.5 * log_var) * noise

        return pred_img, x_start

    @torch.no_grad()
    def ddpm_sample(self, shape):
        # implement the DDPM sampling process.
        img = torch.randn(shape, device=self.device)
        for i in range (0,self.num_timesteps)[::-1]:
            t = torch.full((1,),i, device = self.device, dtype = torch.long)
            img, _ = self.predict_denoised_at_prev_timestep(img, t)

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape):
        # implement the DDIM sampling process.
        img = torch.randn(shape, device=self.device)
        sample_t = torch.linspace(0, self.num_timesteps - 1, steps=self.sampling_timesteps + 1)
        sample_t = torch.flip(sample_t,[0])
        for t in sample_t:
            t = int(t)
            time = torch.full((1,), t, device=self.device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(img, time)
            pre_time = torch.full((1,), t-1, device=self.device, dtype=torch.long)
            if pre_time < 0:
                img = x_start
                continue
            alpha_t = self.alphas_cumprod[t]
            alpha_pre_t = self.alphas_cumprod[t-1]
            sd = torch.sqrt(self.ddim_sampling_eta * ((1 - alpha_t / alpha_pre_t) * (1 - alpha_pre_t) / (1 - alpha_t)))
            noise = torch.randn_like(img)
            img = x_start * torch.sqrt(alpha_pre_t) + torch.sqrt(1 - alpha_pre_t - sd**2)*pred_noise + sd * noise

        img -= torch.min(img)
        img /= torch.max(img)
        return img
