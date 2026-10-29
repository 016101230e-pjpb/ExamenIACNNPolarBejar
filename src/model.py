import torch
import torch.nn as nn

# Semilla para inicialización de pesos
torch.manual_seed(30)

# Pregunta 3.1: Definición de la arquitectura CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Bloque Convolucional 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bloque Convolucional 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Pregunta 3.2: Capa de aplanado (Flatten)
        self.flatten = nn.Flatten()
        
        # Capas Densas (Linear) hacia las 10 clases
        # Dimensiones: 32x32 -> Pool1(16x16) -> Pool2(8x8) | 64 canales * 8 * 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Condición para imprimir shapes intermedios en el paso de validación (Pregunta 3.3)
        debug_mode = not self.training and x.size(0) == 2 
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        if debug_mode: print(f"Shape tras Bloque Conv 1: {x.shape}")
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        if debug_mode: print(f"Shape tras Bloque Conv 2: {x.shape}")
        
        x = self.flatten(x)
        if debug_mode: print(f"Shape tras Flatten: {x.shape}")
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        if debug_mode: print(f"Shape Salida Final: {x.shape}")
        
        return x

if __name__ == '__main__':
    # Pregunta 3.3: Validación de Shapes (Aserción y prueba con lote simulado)
    print("--- Iniciando Validación de Forward Pass y Tensores 4D ---")
    model = SimpleCNN()
    model.eval() # Modo evaluación
    
    # Tensor 4D simulado: (Batch_Size=2, Channels=3, Height=32, Width=32)
    dummy_input = torch.randn(2, 3, 32, 32)
    print(f"Shape Tensor de Entrada: {dummy_input.shape}")
    
    output = model(dummy_input)
    
    # Aserción para validar tamaño de salida
    assert output.shape == (2, 10), f"Error de Shape: Se esperaba (2, 10), pero se obtuvo {output.shape}"
    print("¡Validación exitosa! La CNN procesa correctamente los tensores 4D y la conexión Flatten es exacta.")
