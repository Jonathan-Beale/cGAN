from train import train_gan
from cGAN import Generator, Discriminator
from dataPrep import setup
import torch
import torch.multiprocessing
import time
import pickle

if __name__ == "__main__":
    start = time.time()
    print(f"Setting up data... Timestamp: {time.time()}")
    train_loader, val_loader, device = setup()
    print(f"Data setup complete. Duration: {time.time() - start}")
    # save the loaders

    # start = time.time()
    # print(f"Saving loaders... Timestamp: {time.time()}")
    # with open("loaders/train_loader.pkl", "wb") as f:
    #     pickle.dump(train_loader, f)
    # with open("loaders/val_loader.pkl", "wb") as f:
    #     pickle.dump(val_loader, f)
    # print(f"Loaders saved. Duration: {time.time() - start}")

    start = time.time()
    print(f"Loading loaders... Timestamp: {time.time()}")
    # Load the loaders
    with open("loaders/train_loader.pkl", "rb") as f:
        train_loader = pickle.load(f)
    with open("loaders/val_loader.pkl", "rb") as f:
        val_loader = pickle.load(f)
    print(f"Loaders loaded. Duration: {time.time() - start}")

    start = time.time()
    print(f"Setting up models... Timestamp: {time.time()}")
    generator = Generator(latent_dim=100, num_classes=10).to(device)
    discriminator = Discriminator().to(device)
    print(f"Models setup complete. Duration: {time.time() - start}")

    torch.multiprocessing.freeze_support()  # Optional, safe to include

    results = train_gan(generator, discriminator, train_loader, val_loader, latent_dim=100, device=device)


    # save the results to a file
    import json
    with open("results.json", "w") as f:
        json.dump(results, f)

    # save the model weights with specific names
    generator_path = "models/cGAN_interval_1_5_generator_weights.pth"
    discriminator_path = "models/cGAN_interval_1_5_discriminator_weights.pth"
    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)