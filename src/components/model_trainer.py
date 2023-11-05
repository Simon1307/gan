import torch
import torch.nn as nn
from src.components.data_ingestion import ImageDataset, CombinedDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from src.components.model import Discriminator, Generator
from ..logger import logging
from ..utils import train_fn, save_model, Config


if __name__ == '__main__':
    config = Config()  # Initialize the configuration

    device = config.device
    img_size = config.img_size
    batch_size = config.batch_size
    num_epochs = config.num_epochs
    learning_rate = config.learning_rate
    lambda_cycle = config.lambda_cycle
    lambda_identity = config.lambda_identity
    monet_ds = config.monet_ds
    photo_ds = config.photo_ds
    transformations = config.transforms
    model_path = config.model_path
    logging.info('Loaded the configuration for training')

    monet_tensors = ImageDataset(root=monet_ds, transform=transformations)
    photo_tensors = ImageDataset(root=photo_ds, transform=transformations)
    logging.info('Read the dataset and converted it tensors')
    
    combined_dataset = CombinedDataset(monet_tensors, photo_tensors)
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    logging.info('Created the dataloader')

    # Initializing discriminator P and M
    disc_P = Discriminator(in_channels=3).to(device)  # used for classifying images of photos (fake/real)
    disc_M = Discriminator(in_channels=3).to(device)  # used for classifying images of monets (fake/real)
    # Initializing generators M and P
    gen_M = Generator(img_channels=3, num_residuals=9).to(device)  # used to generate a monet using a photo image as input
    gen_P = Generator(img_channels=3, num_residuals=9).to(device)  # used to generate a photo using a monet image as input

    # print('Output shape of the discriminator:', disc_P(monet_tensors[0].to(device)).shape)  # 1x30x30
    # print('Output shape of the generator:', gen_P(monet_tensors[0].to(device)).shape)  # 3x256x256

    # Initialize optimizer of discriminator using parameters of both discriminators
    opt_disc = optim.Adam(list(disc_P.parameters()) + list(disc_M.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    # Initialize optimizer of generator using parameters of both generators
    opt_gen = optim.Adam(list(gen_M.parameters()) + list(gen_P.parameters()), lr=learning_rate, betas=(0.5, 0.999))

    # L1 loss for cycle consistency loss
    L1 = nn.L1Loss()
    # MSE loss for adversial loss
    mse = nn.MSELoss()

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    logging.info('Initialized discs, gens, optimizers, and losses')
    logging.info('Begin training')
    for epoch in range(num_epochs):
        logging.info('Epoch {}'.format(epoch + 1))
        train_fn(disc_P, disc_M, gen_M, gen_P, dataloader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, epoch, device, lambda_cycle, lambda_identity)

    save_model(gen_M, model_path=model_path)
