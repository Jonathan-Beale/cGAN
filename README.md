Testing novel cGAN architecture using a multi-head discriminator.

## !! Note !!
install torch/torchvision/torchaudio with: 
```
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib opencv-python albumentations
mkdir training loaders models generated
python main.py --latent_dim 256 --epochs 20
v
```