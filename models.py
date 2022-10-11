import torch
import torch.nn as nn
from collections import OrderedDict


class Generator(nn.Module):
    """Your implementation of the generator of DCGAN"""

    def __init__(self, config: dict):
        """TODO: define all layers of the Generator."""
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            OrderedDict([
                # Block 1:input is Z, going into a convolution
                ('ConvTranspose2d_1',
                 None),
                ('BatchNorm2d_1', None),
                ('ReLU_1', None),

                # Block 2: input is (64 * 8) x 4 x 4
                ('ConvTranspose2d_2',
                 None),
                ('BatchNorm2d_2', None),
                ('ReLU_2', None),

                # Block 3: input is (64 * 4) x 8 x 8
                ('ConvTranspose2d_3',
                 None),
                ('BatchNorm2d_3', None),
                ('ReLU_3', None),

                # Block 4: input is (64 * 2) x 16 x 16
                ('ConvTranspose2d_4', None),
                ('BatchNorm2d_4', None),
                ('ReLU_4', None),

                # Block 5: input is (64) x 32 x 32
                ('ConvTranspose2d_5', None),
                ('Tanh', None)
                # Output: output is (3) x 64 x 64
            ])
        )

    def forward(self, input: torch.tensor) -> torch.Tensor:
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    """Your implementation of the discriminator of DCGAN"""

    def __init__(self):
        """TODO: define all layers of the Discriminator."""
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            OrderedDict([
                # Block 1: input is (3) x 64 x 64
                ('Conv2d_1', None),
                ('LeakyReLU_1', None),

                # Block 2: input is (64) x 32 x 32
                ('Conv2d_2', None),
                ('BatchNorm2d_2', None),
                ('LeakyReLU_2', None),

                # Block 3: input is (64*2) x 16 x 16
                ('Conv2d_3', None),
                ('BatchNorm2d_3', None),
                ('LeakyReLU_3', None),

                # Block 4: input is (64*4) x 8 x 8
                ('Conv2d_4', None),
                ('BatchNorm2d_4', None),
                ('LeakyReLU_4', None),

                # Block 5: input is (64*8) x 4 x 4
                ('Conv2d_5', None),
                ('Sigmoid', None),
                ('Flatten', None)
                # Output: 1
            ])
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.main(input)
        return output
