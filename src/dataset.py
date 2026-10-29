import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# Aplicación de semilla individual (Pregunta 1.1)
torch.manual_seed(30)
np.random.seed(30)

def get_data_loaders(batch_size=64):
    # Pregunta 1.3: Pipeline de Transformaciones con Data Augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Pregunta 1.1: Carga del dataset CIFAR-10
    full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Pregunta 1.4: Partición Estratificada (80% train, 20% validation)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    print(f"Dimensiones del dataset de Entrenamiento: {len(train_dataset)} imágenes")
    print(f"Dimensiones del dataset de Validación: {len(val_dataset)} imágenes")
    print(f"Dimensiones del dataset de Prueba: {len(test_dataset)} imágenes")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, full_train_dataset.classes

def show_images(images, labels, classes):
    # Pregunta 1.2: Visualización del Dataset (Cuadrícula con 16 imágenes)
    fig = plt.figure(figsize=(12, 6))
    for i in range(min(16, len(images))):
        ax = fig.add_subplot(2, 8, i+1, xticks=[], yticks=[])
        # Desnormalizar la imagen para visualización
        img = images[i] / 2 + 0.5     
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(classes[labels[i]])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train_loader, _, _, classes = get_data_loaders(batch_size=16)
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    show_images(images, labels, classes)
