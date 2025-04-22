import argparse
import torch
import torch.multiprocessing
import time
import pickle
import json

from train import train_gan
from cGAN import Generator, Discriminator
from dataPrep import setup

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train cGAN with specific latent dimension.")
    parser.add_argument('--latent_dim', type=int, required=True, help='Latent dimension size (e.g., 16, 64, 256)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    args = parser.parse_args()

    latent_dim = args.latent_dim
    epochs = args.epochs

    torch.multiprocessing.freeze_support()  # Windows-safe

    start = time.time()
    print(f"Setting up data... Timestamp: {time.time()}")
    train_loader, val_loader, device = setup()
    print(f"Data setup complete. Duration: {time.time() - start}")

    start = time.time()
    print(f"Saving loaders... Timestamp: {time.time()}")
    with open("loaders/train_loader.pkl", "wb") as f:
        pickle.dump(train_loader, f)
    with open("loaders/val_loader.pkl", "wb") as f:
        pickle.dump(val_loader, f)
    print(f"Loaders saved. Duration: {time.time() - start}")

    print(f"Setting up models with latent_dim = {latent_dim}... Timestamp: {time.time()}")
    generator = Generator(latent_dim=latent_dim, num_classes=10).to(device)
    discriminator = Discriminator().to(device)
    print(f"Models setup complete. Duration: {time.time() - start}")

    print(f"Training GAN with latent dimension {latent_dim} for {epochs} epochs... Timestamp: {time.time()}")
    results = train_gan(generator, discriminator, train_loader, val_loader, latent_dim=latent_dim, device=device, epochs=epochs)
    print(f"GAN training complete. Duration: {time.time() - start}")

    with open(f"results_ld{latent_dim}.json", "w") as f:
        json.dump(results, f)

    generator_path = f"models/cGAN_generator_weights_ld{latent_dim}.pth"
    discriminator_path = f"models/cGAN_discriminator_weights_ld{latent_dim}.pth"
    if isinstance(generator, torch.nn.DataParallel):
        torch.save(generator.module.state_dict(), generator_path)
        torch.save(discriminator.module.state_dict(), discriminator_path)
    else:
        torch.save(generator.state_dict(), generator_path)
        torch.save(discriminator.state_dict(), discriminator_path)
