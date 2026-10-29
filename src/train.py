import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
from dataset import get_data_loaders
from model import SimpleCNN

# Semilla obligatoria
torch.manual_seed(30)
np.random.seed(30)

def train_model():
    os.makedirs('../outputs', exist_ok=True)
    
    train_loader, val_loader, test_loader, classes = get_data_loaders(batch_size=64)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en: {device}")
    
    model = SimpleCNN().to(device)
    
    # Pregunta 4.1: Configuración de Entrenamiento
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    EPOCHS = 10
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # Pregunta 4.2: Bucle de Entrenamiento
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validación
        model.eval()
        val_loss = 0.0
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
                
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct_val / total_val
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Época {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

    # Pregunta 4.3: Generación de Artefactos Gráficos
    print("Generando y guardando gráficas en outputs/...")
    
    # 1. Curvas de Aprendizaje
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS+1), train_losses, marker='o', label='Train Loss')
    plt.plot(range(1, EPOCHS+1), val_losses, marker='o', label='Validation Loss')
    plt.title('Pérdida (Loss) por Época')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS+1), train_accs, marker='o', label='Train Accuracy')
    plt.plot(range(1, EPOCHS+1), val_accs, marker='o', label='Validation Accuracy')
    plt.title('Precisión (Accuracy) por Época')
    plt.xlabel('Época')
    plt.ylabel('Precisión (%)')
    plt.legend()
    
    plt.savefig('../outputs/loss_accuracy.png')
    plt.close()

    # 2. Matriz de Confusión
    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusión - CIFAR-10')
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Real')
    plt.savefig('../outputs/confusion_matrix.png')
    plt.close()
    print("Entrenamiento completado exitosamente.")

if __name__ == '__main__':
    train_model()
