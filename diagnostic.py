import torch
import time
from torch import nn
from torch.utils.data import Dataset, DataLoader

class DummyDataset(Dataset):
    def __len__(self):
        return 8192

    def __getitem__(self, idx):
        image = torch.rand(1, 28, 28)
        label = torch.randint(0, 10, (1,)).item()
        real_flag = 1.0
        return image, label, real_flag

class DummyGen(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 10, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, z, labels):
        one_hot = torch.nn.functional.one_hot(labels, 10).float()
        x = torch.cat([z, one_hot], dim=1)
        return self.net(x).view(-1, 1, 28, 28)

class DummyDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 128), nn.ReLU())
        self.class_out = nn.Linear(128, 10)
        self.real_fake = nn.Linear(128, 1)

    def forward(self, x):
        f = self.features(x)
        return self.class_out(f), self.real_fake(f)

def benchmark_batch_sizes(latent_dim=100, device="cuda"):
    gen = DummyGen(latent_dim).to(device)
    disc = DummyDisc().to(device)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    batch_sizes = [32, 64, 128, 256, 512, 1024, 2048]
    dataset = DummyDataset()

    for bs in batch_sizes:
        loader = DataLoader(dataset, batch_size=bs, num_workers=4, pin_memory=True)

        try:
            torch.cuda.empty_cache()
            gen.eval(); disc.eval()

            it = iter(loader)
            batch = next(it)
            imgs, labels, _ = batch
            imgs = imgs.to(device).float()
            labels = labels.to(device)
            z = torch.randn(bs, latent_dim, device=device)
            fake_labels = torch.randint(0, 10, (bs,), device=device)

            torch.cuda.synchronize()
            t0 = time.time()

            with torch.no_grad():
                fake_imgs = gen(z, fake_labels)
                _, real_out = disc(imgs)
                _, fake_out = disc(fake_imgs)

                d_loss = (
                    bce(real_out, torch.ones_like(real_out)) +
                    bce(fake_out, torch.zeros_like(fake_out))
                )

            torch.cuda.synchronize()
            t1 = time.time()

            elapsed = t1 - t0
            print(f"Batch size: {bs:<5} | Time: {elapsed:.4f}s | Samples/sec: {bs / elapsed:.2f}")

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"❌ OOM at batch size {bs}")
                break
            else:
                raise e

if __name__ == "__main__":
    benchmark_batch_sizes()
