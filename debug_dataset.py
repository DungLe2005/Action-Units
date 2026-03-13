from dataset import *
import matplotlib.pyplot as plt
from config import load_config

params = load_config("config.yaml")

train_loader, val_loader, test_loader, num_classes = create_dataloaders(params)

images, labels = next(iter(train_loader))

print("Image shape:", images.shape)
print("Label:", labels[0])

img = images[0]

mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3,1,1)
std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3,1,1)

img = img * std + mean
img = img.permute(1,2,0).numpy()

plt.imshow(img)
plt.show()