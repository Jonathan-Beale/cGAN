import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # Output: 64x28x28
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # Output: 128x14x14
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # Output: 256x7x7
        self.fc = nn.Linear(256 * 7 * 7, 512)

        # Two separate output layers
        self.digit_head = nn.Linear(512, 10)  # Classifies digits 0-9
        self.real_fake_head = nn.Linear(512, 1)  # Determines real vs fake\
    def forward(self, x):
        x = x.to(next(self.parameters()).device)  # Ensure x is on the same device as model parameters
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = x.view(x.shape[0], -1)  # Flatten
        x = F.leaky_relu(self.fc(x), 0.2)

        digit_output = self.digit_head(x)  # Remove log_softmax
        real_fake_output = torch.sigmoid(self.real_fake_head(x))  # Real/Fake classification
        return digit_output, real_fake_output

class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Embedding layer for digit labels
        self.label_emb = nn.Embedding(num_classes, num_classes)

        # Fully connected layer to project noise + label into feature space
        self.fc = nn.Linear(latent_dim + num_classes, 256 * 7 * 7)

        # Deconvolution layers to generate 28x28 image
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1)  # Output: 1x28x28

    def forward(self, z, labels):
        z = z.to(next(self.parameters()).device)  # Move z to model device
        labels = labels.to(z.device)  # Ensure labels match the device of z
        assert labels.min() >= 0 and labels.max() < 10, f"Generator Label Error! Min: {labels.min()}, Max: {labels.max()}"


        label_embedding = self.label_emb(labels)  # Labels must be on the same device as embedding
        x = torch.cat((z, label_embedding), dim=1)  # Concatenate noise and label
        x = F.leaky_relu(self.fc(x), 0.2)
        x = x.view(-1, 256, 7, 7)  # Reshape for deconvolutions

        x = F.leaky_relu(self.deconv1(x), 0.2)
        x = F.leaky_relu(self.deconv2(x), 0.2)
        x = torch.tanh(self.deconv3(x))  # Output image in range [-1, 1]
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate test set
def generate_test_set(generator, num_samples, latent_dim, device):
    z = torch.randn(num_samples, latent_dim, device=device)
    labels = torch.randint(0, 10, (num_samples,), device=device)
    real_authentic = torch.zeros(num_samples, 1, device=device)  # Fake images
    with torch.no_grad():
        generated_images = generator(z, labels)
    return generated_images, labels, real_authentic


def compute_classification_accuracy(discriminator, dataloader, device):
    discriminator.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, real_flag in dataloader:
            real_indices = (real_flag.view(-1) == 1.0)
            if real_indices.sum() == 0:
                continue  # skip if no real images in batch

            images = images[real_indices].to(device)
            labels = labels[real_indices].to(device)

            outputs, _ = discriminator(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0

def compute_discrimination_accuracy(discriminator, dataloader, device, threshold=0.5):
    """
    Computes binary accuracy of discriminator for real vs fake images.
    
    Args:
        discriminator: The discriminator model.
        dataloader: Dataloader providing (image, label, real_flag).
        device: torch.device('cuda' or 'cpu').
        threshold: Threshold to binarize real/fake logits or probs (default 0.5).
    
    Returns:
        Float between 0 and 1 representing binary accuracy.
    """
    discriminator.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, real_flag in dataloader:
            images = images.to(device)
            real_flag = real_flag.view(-1).to(device)  # Shape: (batch_size,)

            # Discriminator returns (class_output, real_fake_output)
            _, outputs = discriminator(images)

            # If output is (batch_size, 1), squeeze it
            if outputs.size(1) == 1:
                outputs = outputs.view(-1)

            predicted = (outputs >= threshold).float()
            correct += (predicted == real_flag).sum().item()
            total += real_flag.size(0)

    return correct / total if total > 0 else 0.0
