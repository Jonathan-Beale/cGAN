
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SubsetRandomSampler
import time
from cGAN import generate_test_set, compute_classification_accuracy, compute_discrimination_accuracy

class FakeEvalDataset(Dataset):
    def __init__(self, images, labels, flags):
        self.images = images
        self.labels = labels
        self.flags = flags

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.flags[idx]

# def train_gan(generator, discriminator, train_loader, val_loader, latent_dim, device, epochs=100, gen_interval=1, disc_interval=5):
#     train_metrics = {"train_acc": [], "val_acc": [], "fool_rate": [], "completion_time": []}
#     # send train_loader to device
#     for images, labels, real_flag in train_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         real_flag = real_flag.to(device)
    
#     # send val_loader to device
#     for images, labels, real_flag in val_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         real_flag = real_flag.to(device)

#     # Fixed 20% validation split
#     # Initialize optimizers and loss
#     gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#     disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#     criterion = nn.BCELoss()
#     criterion_class = nn.CrossEntropyLoss()

#     print(f"Generator device: {next(generator.parameters()).device}")
#     print(f"Discriminator device: {next(discriminator.parameters()).device}")

#     mode = "gen"
#     mode_epoch_counter = 0
#     start_time = time.time()

#     for epoch in range(epochs):
#         # Handle fixed-rhythm mode switching
#         if (mode == "gen" and mode_epoch_counter >= gen_interval) or (mode == "disc" and mode_epoch_counter >= disc_interval):
#             mode = "disc" if mode == "gen" else "gen"
#             mode_epoch_counter = 0
        
#         generated_eval_set = []

#         mode_epoch_counter += 1

#         total_gen_loss = 0
#         total_disc_loss = 0
#         total_disc_acc = 0
#         total_fool_rate = 0
#         batch_count = len(train_loader)
#         gen_step_count = 0

#         for batch_idx, (real_images, real_labels, real_authentic) in enumerate(train_loader):
#             real_images = real_images.to(device)
#             real_labels = real_labels.to(device)
#             real_authentic = real_authentic.view(-1, 1).to(device)

#             # --- Discriminator Step ---
#             discriminator.train()
#             fake_images, fake_labels, fake_authentic = generate_test_set(generator, real_images.size(0), latent_dim, device)
#             real_class_output, real_output = discriminator(real_images)
#             fake_class_output, fake_output = discriminator(fake_images)

#             real_loss = criterion(real_output, real_authentic)
#             fake_loss = criterion(fake_output, fake_authentic)
#             class_loss = criterion_class(real_class_output, real_labels)
#             loss_disc = real_loss + fake_loss + class_loss

#             disc_optimizer.zero_grad()
#             loss_disc.backward()
#             disc_optimizer.step()
#             total_disc_loss += loss_disc.item()

#             real_correct = (real_output > 0.7).float().mean().item()
#             fake_correct = (fake_output < 0.3).float().mean().item()
#             disc_acc = (real_correct + fake_correct) / 2
#             total_disc_acc += disc_acc

#             # --- Generator Step (every `gen_interval` batches) ---
#             if gen_step_count % gen_interval == 0:
#                 generator.train()
#                 z = torch.randn(real_images.size(0), latent_dim, device=device)
#                 fake_labels = torch.randint(0, 10, (real_images.size(0),), device=device)
#                 fake_images = generator(z, fake_labels)
#                 fake_authentic = torch.ones(real_images.size(0), 1, device=device)

#                 class_output, disc_output = discriminator(fake_images)
#                 loss_gen_class = criterion_class(class_output, fake_labels)
#                 loss_gen = criterion(disc_output, fake_authentic) + loss_gen_class

#                 gen_optimizer.zero_grad()
#                 loss_gen.backward()
#                 gen_optimizer.step()
#                 total_gen_loss += loss_gen.item()

#                 fool_rate = (disc_output > 0.5).float().mean().item()
#                 total_fool_rate += fool_rate

#                 generated_eval_set.append((
#                     fake_images.detach().cpu(),
#                     fake_labels.detach().cpu(),
#                     torch.zeros_like(fake_authentic).detach().cpu()
#                 ))

#             gen_step_count += 1

#             # --- Logging ---
#             if (batch_idx + 1) % max(1, (batch_count // 20)) == 0:
#                 progress = (batch_idx + 1) / batch_count * 100
#                 print(f"Epoch {epoch+1}/{epochs} - {progress:.1f}% | Time: {time.time()} | Fool Rate: {fool_rate or None}| Mode: {mode} | Gen Loss: {total_gen_loss:.4f} | Disc Loss: {total_disc_loss:.4f}")

#         # Evaluation
#         train_acc = compute_classification_accuracy(discriminator, train_loader, device)
#         val_acc = compute_classification_accuracy(discriminator, val_loader, device)
#         if generated_eval_set:
#             all_images = torch.cat([x[0] for x in generated_eval_set], dim=0)
#             all_labels = torch.cat([x[1] for x in generated_eval_set], dim=0)
#             all_flags  = torch.cat([x[2] for x in generated_eval_set], dim=0)

#             fake_eval_loader = DataLoader(FakeEvalDataset(all_images, all_labels, all_flags),
#                                         batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
            
#             # Fool rate = % of fake images predicted as real
#             fool_rate = compute_discrimination_accuracy(discriminator, fake_eval_loader, device)
#         else:
#             fool_rate = 0.0  # If we didn’t run gen mode this epoch

#         train_metrics["train_acc"].append(train_acc)
#         train_metrics["val_acc"].append(val_acc)
#         train_metrics["fool_rate"].append(fool_rate)
#         train_metrics["completion_time"].append(time.time() - start_time)

#         print(f"Epoch {epoch+1} Summary - Mode: {mode} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Fool Rate: {fool_rate:.4f}")

#         with torch.no_grad():
#             test_images, test_labels, _ = generate_test_set(generator, 25, latent_dim, device)
#             class_preds, real_fake_preds = discriminator(test_images)
#             class_preds = class_preds.argmax(dim=1)
#             real_fake_preds = real_fake_preds.squeeze().cpu().numpy()

#         save_image(test_images, f"generated/generated_epoch_{epoch+1}_dim_{latent_dim}.png", nrow=5, normalize=True)

#     return train_metrics


# Training loop
def train_gan(generator, discriminator, train_loader, val_loader, latent_dim, device, epochs=100):
    train_metrics = {"train_acc": [], "val_acc": [], "fool_rate": [], "completion_time": []}
    
    # Initialize optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    criterion_class = nn.CrossEntropyLoss()

    print(f"Generator device: {next(generator.parameters()).device}")
    print(f"Discriminator device: {next(discriminator.parameters()).device}")
    
    mode = "gen"  # Start in generator mode

    for epoch in range(epochs):
        epoch_start = time.time()
        total_gen_loss = 0
        total_disc_loss = 0
        total_disc_acc = 0
        total_fool_rate = 0
        batch_count = len(train_loader)

        for batch_idx, (real_images, real_labels, real_authentic) in enumerate(train_loader):
            batch_start = time.time()
            real_images, real_labels, real_authentic = (
                real_images.to(device),
                real_labels.to(device),
                real_authentic.view(-1, 1).to(device),
            )

            if mode == "gen":
                # ---------------------
                #  Train Generator
                # ---------------------
                generator.train()
                z = torch.randn(real_images.size(0), latent_dim, device=device)
                fake_labels = torch.randint(0, 10, (real_images.size(0),), device=device)
                fake_images = generator(z, fake_labels)
                fake_authentic = torch.ones(real_images.size(0), 1, device=device)  # Generator wants to fool discriminator

                class_output, disc_output = discriminator(fake_images)
                loss_gen_class = criterion_class(class_output, fake_labels)  # Encourage class-aware generation
                loss_gen = criterion(disc_output, fake_authentic) + loss_gen_class

                gen_optimizer.zero_grad()
                loss_gen.backward()
                gen_optimizer.step()
                total_gen_loss += loss_gen.item()

                # Compute fool rate (how often generator fools the discriminator)
                fool_rate = (disc_output > 0.5).float().mean().item()
                total_fool_rate += fool_rate

                # Swap to Discriminator Mode if fool rate > 90%
                if fool_rate > 0.9:
                    mode = "disc"

            else:
                # ---------------------
                #  Train Discriminator
                # ---------------------
                discriminator.train()
                fake_images, fake_labels, fake_authentic = generate_test_set(generator, real_images.size(0), latent_dim, device)
                real_class_output, real_output = discriminator(real_images)
                fake_class_output, fake_output = discriminator(fake_images)

                real_loss = criterion(real_output, real_authentic)
                fake_loss = criterion(fake_output, fake_authentic)
                class_loss = criterion_class(real_class_output, real_labels)
                loss_disc = real_loss + fake_loss + class_loss

                disc_optimizer.zero_grad()
                loss_disc.backward()
                disc_optimizer.step()
                total_disc_loss += loss_disc.item()

                # Compute discriminator accuracy
                real_correct = (real_output > 0.7).float().mean().item()
                fake_correct = (fake_output < 0.3).float().mean().item()
                disc_acc = (real_correct + fake_correct) / 2
                total_disc_acc += disc_acc

                # Swap to Generator Mode if discriminator accuracy > 50%
                if disc_acc > 0.5:
                    mode = "gen"

            # Output every 5% of epoch
            if (batch_idx + 1) % (batch_count // 20) == 0:
                progress = (batch_idx + 1) / batch_count * 100
                print(f"Epoch {epoch+1}/{epochs} - {progress:.1f}% | Mode: {mode} | Completion Time {time.time() - batch_start:.4f} | Gen Loss: {loss_gen:.4f} | Disc Loss: {loss_disc:.4f} | Disc Acc: {disc_acc:.4f}")

        # ---------------------
        #  End of Epoch Logging & Visualization
        # ---------------------
        # Compute classification accuracy of discriminator on real training images
        start_time = time.time()
        print(f"Evaluating discriminator...")
        train_acc = compute_classification_accuracy(discriminator, train_loader, device)
        val_acc = compute_classification_accuracy(discriminator, val_loader, device)
        avg_fool_rate = total_fool_rate / batch_count
        print(f"Completion Time: {time.time() - start_time:.4f}")

        # Store metrics
        train_metrics["train_acc"].append(train_acc)
        train_metrics["val_acc"].append(val_acc)
        train_metrics["fool_rate"].append(avg_fool_rate)
        train_metrics["completion_time"].append(time.time() - start_time)

        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Avg Fool Rate: {avg_fool_rate:.4f} | Completion Time: {time.time() - epoch_start:.4f}")
        # Generate example images
        with torch.no_grad():
            test_images, test_labels, _ = generate_test_set(generator, 25, latent_dim, device)

            # Get discriminator predictions
            class_preds, real_fake_preds = discriminator(test_images)
            class_preds = class_preds.argmax(dim=1)  # Get predicted class
            real_fake_preds = real_fake_preds.squeeze().cpu().numpy()  # Convert to NumPy

        # Save and visualize generated images
        save_image(test_images, f"generated/generated_epoch_{epoch+1}_dim_{latent_dim}.png", nrow=5, normalize=True)
        # visualize_augmentations(test_images, test_labels, class_preds, real_fake_preds, num_images=5, epoch=epoch+1, latent_dim=latent_dim)
    return train_metrics
