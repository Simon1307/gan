import torch
import os
import random
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from src.logger import logging


def save_model(model, model_path):
    torch.save(model, model_path)


def load_model(model_path, device):
    model = torch.load('artifacts/gen_M.pth', map_location=device)
    return model


class Config:
    def __init__(self):
        # General Configuration
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.img_size = 256
        self.batch_size = 4
        self.num_epochs = 1
        self.learning_rate = 1e-5
        self.lambda_identity = 0.0
        self.lambda_cycle = 10

        # Dataset Paths
        self.monet_ds = 'gan-getting-started/monet_jpg_small'
        self.photo_ds = 'gan-getting-started/photo_jpg_small'

        # Transforms
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        self.pred_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        # Model Paths
        self.model_path = 'artifacts/gen_M.pth'


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_fn(disc_P, disc_M, gen_M, gen_P, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch, device, lambda_cycle, lambda_identity):
    P_reals = 0
    P_fakes = 0

    G_loss_total = 0.0
    D_loss_total = 0.0

    # Loop has the length of images in loader, to get a progress bar
    loop = tqdm(loader, leave=True)

    # One epoch is done when every image is used once
    # Per iteration one image is used as input since batch_size=1
    #for idx, (monet, photo) in enumerate(loop):

    for idx, batch in enumerate(tqdm(loader, desc=f'Epoch {epoch}')):
        # Inside this loop, 'batch' contains pairs of images from domain A and B
        monet = batch['A'].to(device)
        photo = batch['B'].to(device)


        # Train Discriminators P and M
        with torch.cuda.amp.autocast():  # allows the script to run in mixed precision (float32 and float16 for example)
            fake_photo = gen_P(monet)  # generator generates fake photo
            D_P_real = disc_P(photo)  # discriminator classifies a real photo image, size: 1,1,30,30
            D_P_fake = disc_P(fake_photo.detach())  # discriminator classifies a fake photo image, size: 1,1,30,30

            P_reals += D_P_real.mean().item()
            P_fakes += D_P_fake.mean().item()

            # MSE of real and fake photo image (1=real, 0=fake)
            D_P_real_loss = mse(D_P_real, torch.ones_like(D_P_real))
            D_P_fake_loss = mse(D_P_fake, torch.zeros_like(D_P_fake))

            # Combined MSE loss of discriminator P
            D_P_loss = D_P_real_loss + D_P_fake_loss

            fake_monet = gen_M(photo)  # generator generates fake monet
            D_M_real = disc_M(monet)  # discriminator classifies a real monet image
            D_M_fake = disc_M(fake_monet.detach())  # discriminator classifies a fake monet image

            # MSE of real and fake monet image
            D_M_real_loss = mse(D_M_real, torch.ones_like(D_M_real))
            D_M_fake_loss = mse(D_M_fake, torch.zeros_like(D_M_fake))

            # Combined MSE loss of discriminator M
            D_M_loss = D_M_real_loss + D_M_fake_loss

            # Combined MSE loss of discriminators P and M
            D_loss = (D_P_loss + D_M_loss) / 2

        # Sum discrimintator loss per epoch
        D_loss_total += D_loss.item()

        opt_disc.zero_grad()
        # backpropagation
        d_scaler.scale(D_loss).backward()
        # Parameter update based on the current gradient
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators P and M
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_P_fake = disc_P(fake_photo)
            D_M_fake = disc_M(fake_monet)
            # generator wants to fool the discriminator
            loss_G_P = mse(D_P_fake, torch.ones_like(D_P_fake))
            loss_G_M = mse(D_M_fake, torch.ones_like(D_M_fake))

            # cycle loss
            # generating the original monet using the generated fake photo
            cycle_monet = gen_M(fake_photo)
            # generating the original photo using the generated fake monet
            cycle_photo = gen_P(fake_monet)
            # Compute l1 loss with original monet and generated original monet
            cycle_monet_loss = l1(monet, cycle_monet)
            # Compute l1 loss with original photo and generated original photo
            cycle_photo_loss = l1(photo, cycle_photo)

            # identity loss
            # identity_monet = gen_M(monet)
            # identity_photo = gen_P(photo)
            # identity_monet_loss = l1(monet, identity_monet)
            # identity_photo_loss = l1(photo, identity_photo)

            # Combined MSE loss of generators P and M
            G_loss = (
                    loss_G_M
                    + loss_G_P
                    + cycle_monet_loss * lambda_cycle
                    + cycle_photo_loss * lambda_cycle
                # + identity_photo_loss * config.LAMBDA_IDENTITY
                # + identity_monet_loss * config.LAMBDA_IDENTITY
            )

        # Sum generators loss per epoch
        G_loss_total += G_loss.item()

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # Update progress bar
        loop.set_postfix(P_real=P_reals / (idx + 1), P_fake=P_fakes / (idx + 1))
    logging.info('G_loss_total: {}'.format(G_loss_total))
    logging.info('D_loss_total: {}'.format(D_loss_total))