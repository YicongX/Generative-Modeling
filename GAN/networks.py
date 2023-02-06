import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F



class UpSampleConv2D(jit.ScriptModule):
    # Implement nearest neighbor upsampling + conv layer

    def __init__(
        self,
        input_channels,
        kernel_size=3,
        n_filters=128,
        upscale_factor=2,
        padding=0,
    ):
        super(UpSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(input_channels, n_filters, kernel_size,padding = padding)
        self.upscale_factor = upscale_factor
        self.pix_shuffle = nn.PixelShuffle(self.upscale_factor)
        

    @jit.script_method
    def forward(self, x):
        # Implement nearest neighbor upsampling.
        # 1. Duplicate x channel wise upscale_factor^2 times.
        # 2. Then re-arrange to form an image of shape (batch x channel x height*upscale_factor x width*upscale_factor).
        # 3. Apply convolution.

        # trans = nn.Upsample(scale_factor=self.upscale_factor, mode='nearest')
        # x = trans(x)

        x_out = x.repeat(1,self.upscale_factor*self.upscale_factor,1,1)
        x_out = self.pix_shuffle(x_out)
        x_out = self.conv(x_out)
        return x_out


class DownSampleConv2D(jit.ScriptModule):
    # Implement spatial mean pooling + conv layer

    def __init__(
        self, input_channels, kernel_size=3, n_filters=128, downscale_ratio=2, padding=0
    ):
        super(DownSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(input_channels, n_filters, kernel_size,padding = padding)
        self.downscale_ratio = downscale_ratio
        self.pix_unshuffle = nn.PixelUnshuffle(self.downscale_ratio)

    @jit.script_method
    def forward(self, x):
        # Implement spatial mean pooling.
        # 1. Re-arrange to form an image of shape: (batch x channel * upscale_factor^2 x height x width).
        # 2. Then split channel wise into upscale_factor^2 number of images of shape: (batch x channel x height x width).
        # 3. Average the images into one and apply convolution.
        
        x_out = self.pix_unshuffle(x)
        x_out = x_out.reshape(4,x_out.shape[0],-1,x_out.shape[2],x_out.shape[3])
        x_out = torch.mean(x_out,dim=0)
        x_out = self.conv(x_out)
        
        return x_out


class ResBlockUp(jit.ScriptModule):
    # Impement Residual Block Upsampler.
    """
    ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(in_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockUp, self).__init__()
        self.res_up = nn.Sequential(
            nn.BatchNorm2d(input_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size, stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.residual = UpSampleConv2D(n_filters, kernel_size, n_filters,padding = 1)
        self.shortcut = UpSampleConv2D(input_channels, kernel_size = 1)

    @jit.script_method
    def forward(self, x):
        # Forward through the layers and implement a residual connection.
        # Apply self.residual to the output of self.layers and apply self.shortcut to the original input.
        x_in = x
        x = self.res_up(x)
        x = self.residual(x)
        return self.shortcut(x_in) + x


class ResBlockDown(jit.ScriptModule):
    # Impement Residual Block Downsampler.
    """
    ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
        )
        (residual): DownSampleConv2D(
            (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): DownSampleConv2D(
            (conv): Conv2d(in_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockDown, self).__init__()
        self.res_down = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size,stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )
        self.residual = DownSampleConv2D(n_filters, kernel_size, n_filters,padding = 1)
        self.shortcut = DownSampleConv2D(input_channels, kernel_size = 1)

    @jit.script_method
    def forward(self, x):
        # Forward through the layers and implement a residual connection.
        # Apply self.residual to the output of self.layers and apply self.shortcut to the original input.
        x_in = x
        x = self.res_down(x)
        x = self.residual(x)
        return self.shortcut(x_in) + x
        

class ResBlock(jit.ScriptModule):
    # Impement Residual Block as described below.
    """
    ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size, stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size, stride=(1, 1), padding=(1, 1)),
        )

    @jit.script_method
    def forward(self, x):
        # Forward the conv layers. Don't forget the residual connection!
        x_in = x
        x = self.layers(x)
        return x + x_in


class Generator(jit.ScriptModule):
    # Impement Generator. Follow the architecture described below:
    """
    Generator(
    (dense): Linear(in_features=128, out_features=2048, bias=True)
    (layers): Sequential(
        (0): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockUp(
            (layers): Sequential(
                (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (1): ReLU()
                (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlockUp(
            (layers): Sequential(
                (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (1): ReLU()
                (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU()
        (5): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): Tanh()
    )
    )
    """

    def __init__(self, starting_image_size=4):
        super(Generator, self).__init__()
        self.dense = nn.Linear(128,2048,bias=True)
        self.model = nn.Sequential(
            ResBlockUp(input_channels = 128, kernel_size = 3, n_filters = 128),
            ResBlockUp(input_channels = 128, kernel_size = 3, n_filters = 128),
            ResBlockUp(input_channels = 128, kernel_size = 3, n_filters = 128),
            nn.BatchNorm2d(128, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Tanh(),
        )

    @jit.script_method
    def forward_given_samples(self, z):
        # forward the generator assuming a set of samples z have been passed in.
        z = self.dense(z).reshape(-1,128,4,4)
        z = self.model(z)
        return z

    @jit.script_method
    def forward(self, n_samples: int = 1024):
        # Generate n_samples latents and forward through the network.
        
        z = torch.randn(n_samples, 128, dtype = torch.half).cuda()
        z = self.forward_given_samples(z)
        return z
        
        


class Discriminator(jit.ScriptModule):
    # Impement Discriminator. Follow the architecture described below:
    """
    Discriminator(
    (layers): Sequential(
        (0): ResBlockDown(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
            )
            (residual): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (shortcut): DownSampleConv2D(
                (conv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
            )
        )
        (1): ResBlockDown(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
            )
            (residual): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (shortcut): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            )
        )
        (2): ResBlock(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
                (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (3): ResBlock(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
                (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (4): ReLU()
    )
    (dense): Linear(in_features=128, out_features=1, bias=True)
    )
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            ResBlockDown(input_channels = 3, kernel_size = 3, n_filters = 128),
            ResBlockDown(input_channels = 128, kernel_size = 3, n_filters = 128),
            ResBlock(input_channels = 128, kernel_size = 3, n_filters = 128),
            ResBlock(input_channels = 128, kernel_size = 3, n_filters = 128),
            nn.ReLU(),
        )
        self.dense = nn.Linear(128,1)

    @jit.script_method
    def forward(self, x):
        # Forward the discriminator assuming a batch of images have been passed in.
        x = self.model(x)
        x = torch.sum(x, dim = (2,3))
        x = self.dense(x)

        return x
