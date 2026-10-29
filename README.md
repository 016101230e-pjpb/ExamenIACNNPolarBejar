# Evaluación de la Segunda Unidad IA - CNN sobre CIFAR-10

**Estudiante:** Patrik Joshua Polar Bejar  
**Código de Matrícula / Semilla:** 016101230E / 30  

Este repositorio contiene el desarrollo del examen práctico sobre Clasificación de Imágenes mediante Redes Neuronales Convolucionales (CNN) y Análisis de Métricas usando PyTorch, de acuerdo a la estructura obligatoria requerida.

## Pregunta 2: Diseño Arquitectónico y Justificación Estructural (CNN vs MLP)

### 1. Justificación Teórica: ¿Por qué un MLP colapsaría?
Un Perceptrón Multicapa (MLP) estándar sería altamente ineficiente y propenso al colapso al procesar las matrices de píxeles de CIFAR-10 (32x32x3) por las siguientes razones:
- **Explosión de Parámetros:** Un MLP requiere "aplanar" la imagen a un vector 1D de 3072 elementos. Una sola capa oculta inicial con 1000 neuronas generaría más de 3 millones de parámetros, lo cual sobrecarga el cómputo y lleva casi inevitablemente al *overfitting* (sobreajuste).
- **Pérdida de Jerarquía Espacial:** Al aplanar la imagen, el MLP destruye la relación espacial 2D entre píxeles vecinos. 
- **Solución de las CNN:** Las CNN solucionan esto mediante **Conexiones Locales** (cada neurona procesa solo un "parche" de la imagen) y **Weight Sharing (Compartición de Pesos)**, usando el mismo filtro convolucional a lo largo de toda la imagen. Esto reduce drásticamente los parámetros y garantiza la **Invariancia Espacial** (capacidad de reconocer un patrón sin importar en qué parte de la imagen se encuentre).

### 2. Especificación de la Arquitectura
El modelo implementado en `src/model.py` incluye:
- **Bloque Convolucional 1:** `Conv2d` (3 canales entrada, 32 salida, kernel 3x3, padding 1) $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d` (kernel 2x2). *(Dimensiones de salida: 32x16x16)*
- **Bloque Convolucional 2:** `Conv2d` (32 canales entrada, 64 salida, kernel 3x3, padding 1) $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d` (kernel 2x2). *(Dimensiones de salida: 64x8x8)*
- **Capa de Aplanado (Flatten):** Transforma el tensor 4D de (Batch, 64, 8, 8) en un vector 1D de 4096 características.
- **Clasificador Denso (Linear):** 
  - Capa Densa 1: 4096 $\rightarrow$ 512 neuronas + `ReLU`.
  - Capa Densa 2 (Salida): 512 $\rightarrow$ 10 neuronas (Clases finales).

## Pregunta 4: Análisis Crítico de Métricas (Semilla 30)

  Basado en las curvas generadas (ver `outputs/loss_accuracy.png`), se puede observar el comportamiento del modelo optimizado con Adam y Loss de CrossEntropy:
  - **Trayectoria de las Curvas:** La pérdida de entrenamiento (Train Loss) decrece progresivamente épocaa a época. La precisión (Accuracy) muestra una mejora continua, demostrando que la red es capaz de aprender los patrones.
  - **Diagnóstico de Overfitting / Underfitting:** 
    - Gracias a la aplicación de **Data Augmentation** (RandomCrop, RandomHorizontalFlip) en el pipeline de transformaciones (`src/dataset.py`), la brecha entre las curvas de entrenamiento y validación se mantiene controlada. 
    - Si en las últimas épocas la *Validation Loss* se estanca o muestra ligeros incrementos mientras que la *Train Loss* sigue bajando, estamos ante síntomas tempranos de **overfitting**, lo cual es esperado en modelos sin regularización de Dropout. Sin embargo, el modelo generaliza razonablemente bien sobre los datos invisibles.
  - **Matriz de Confusión:** Al observar `outputs/confusion_matrix.png`, se nota una fuerte diagonal principal, aunque las clases que comparten texturas y formas similares (como *gatos* y *perros*, o *aviones* y *pájaros*) concentran la mayor cantidad de falsos positivos y negativos.

## ¿Cómo ejecutar el proyecto?
1. Instalar dependencias: `pip install -r requirements.txt`
2. Validar datos y formas: `python src/dataset.py` y `python src/model.py`
3. Entrenar el modelo y generar outputs: `python src/train.py`
