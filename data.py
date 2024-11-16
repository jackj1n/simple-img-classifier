from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchvision

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download CIFAR-10 dataset
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

# Confirm the data is loaded correctly
# print(f"Training samples: {len(train_set)}, Testing samples: {len(test_set)}")

# # Un-normalize and display an image
# def imshow(img):
#     img = img / 2 + 0.5  # Unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0))) # Convert from (C, H, W) to (H, W, C)
#     plt.show()

# # Get some random training images
# data_iter = iter(train_loader)
# images, labels = next(data_iter)

# # Show images and labels
# # Labels: 0	Airplane, 1	Automobile, 2 Bird, 3 Cat, 4 Deer, 5 Dog, 6 Frog, 7 Horse , 8 Ship, 9 Truck
# print('Labels:', ' '.join(str(labels[j].item()) for j in range(8)))  # Show first 8 labels
# imshow(torchvision.utils.make_grid(images))
