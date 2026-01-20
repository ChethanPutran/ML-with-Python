import torch
from torch import nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST  # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def generate_generator_block(infeatures, ouput_size):
    '''
    Function for returning a block of the generator's neural network
    given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a generator neural network layer, with a linear transformation 
          followed by a batch normalization and then a relu activation
    '''
    return nn.Sequential(
        nn.Linear(infeatures, ouput_size),
        nn.BatchNorm1d(ouput_size),
        nn.ReLU(inplace=True)
    )


def get_noise_vector(n_samples, img_size, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    return torch.randn((n_samples, img_size), device=device)


def generate_discriminator_block(input_dim, output_dim):
    '''
    Discriminator Block
    Function for returning a neural network of the discriminator given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a discriminator neural network layer, with a linear transformation 
          followed by an nn.LeakyReLU activation with negative slope of 0.2 
    '''
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(negative_slope=0.2)
    )


class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
          (MNIST images are 28 x 28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, noise_size, img_size, hidden_size):
        super().__init__()
        self.gen = nn.Sequential(
            generate_generator_block(noise_size, hidden_size),
            generate_generator_block(hidden_size, 2*hidden_size),
            generate_generator_block(2*hidden_size, 4*hidden_size),
            generate_generator_block(4*hidden_size, 8*hidden_size),
            nn.Linear(8*hidden_size, img_size),
            nn.Sigmoid()
        )
    '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''

    def forward(self, noise):
        return self.gen(noise)


class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
            (MNIST images are 28x28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, img_dim, hidden_dim):
        super().__init__()
        self.disrim = nn.Sequential(
            generate_discriminator_block(img_dim, hidden_dim*4),
            generate_discriminator_block(hidden_dim*4, hidden_dim*2),
            generate_discriminator_block(hidden_dim*2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, img):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        '''
        return self.disrim(img)


def get_generator_loss(gen, disc, criterion, num_images, img_size, device):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        gen_loss: a torch scalar loss value for the current batch
    '''

    # Create noise vectors
    noise = get_noise_vector(num_images, img_size, device)
    gen_out = gen(noise)  # Only disciminator is updated

    disc_out_fake = disc(gen_out)
    gt = torch.ones_like(disc_out_fake)  # Fooling

    gen_loss = criterion(disc_out_fake, gt)
    return gen_loss


def get_discriminator_loss(gen, disc, criterion, real, num_images, img_size, device):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        real: a batch of real images
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        img_size: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    '''

    # Create noise vectors
    noise = get_noise_vector(num_images, img_size, device)
    gen_out = gen(noise).detach()  # Only disciminator is updated

    disc_out_fake = disc(gen_out)
    fake_gt = torch.zeros_like(disc_out_fake)
    fake_loss = criterion(disc_out_fake, fake_gt)

    disc_out_real = disc(real)
    real_gt = torch.ones_like(disc_out_real)
    real_loss = criterion(disc_out_real, real_gt)

    disc_loss = (real_loss + fake_loss)/2
    return disc_loss


criterion = nn.BCEWithLogitsLoss()
n_epochs = 3
img_dim = 784  # 24*24
display_step = 500
batch_size = 128
lr = 0.00001
noise_dim = 64
hidden_dim = 128

# Load MNIST dataset as tensors
dataloader = DataLoader(
    MNIST('datasets/',
          download=False, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True)

device = 'cuda'

gen = Generator(noise_size=noise_dim, img_size=img_dim,
                hidden_size=hidden_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator(img_dim=img_dim, hidden_dim=hidden_dim).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

# Training
cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
test_generator = True  # Whether the generator should be tested
gen_loss = False
error = False
for epoch in range(n_epochs):

    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)

        # Flatten the batch of real images from the dataset
        real = real.view(cur_batch_size, -1).to(device)

        ### Update discriminator ###
        # Zero out the gradients before backpropagation
        disc_opt.zero_grad()

        # Calculate discriminator loss
        disc_loss = get_discriminator_loss(
            gen, disc, criterion, real, cur_batch_size, noise_dim, device)

        # Update gradients
        disc_loss.backward(retain_graph=True)

        # Update optimizer
        disc_opt.step()

        # For testing purposes, to keep track of the generator weights
        if test_generator:
            old_generator_weights = gen.gen[0][0].weight.detach().clone()

        gen_opt.zero_grad()

        # Calculate discriminator loss
        gen_loss = get_generator_loss(
            gen, disc, criterion, cur_batch_size, noise_dim, device)

        # Update gradients
        gen_loss.backward(retain_graph=True)

        # Update optimizer
        gen_opt.step()

        # For testing purposes, to check that your code changes the generator weights
        if test_generator:
            try:
                assert lr > 0.0000002 or (
                    gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                assert torch.any(
                    gen.gen[0][0].weight.detach().clone() != old_generator_weights)
            except:
                error = True
                print("Runtime tests have failed")

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step

        ### Visualization code ###
        if cur_step % display_step == 0 and cur_step > 0:
            print(
                f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            fake_noise = get_noise_vector(
                cur_batch_size, noise_dim, device=device)
            fake = gen(fake_noise)
            show_tensor_images(fake)
            show_tensor_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1
