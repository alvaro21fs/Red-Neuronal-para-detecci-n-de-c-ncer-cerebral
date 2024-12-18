<header>
</header>

# Proyecto para detección de tumor cerebral
## Introducción
Se ha utilizado el dataset de Kaggle llamado "Brain Tumor MRI Dataset". Cabe mencionar, que este conjunto de datos se divide en cuatro categorías, una perteneciente a no tumor y otras tres que son distintos tipos de tumores (class_names = ['notumor', 'meningioma', 'glioma', 'pituitary']).

Con las citadas imágenes se han entrenado varios modelos de redes neuronales obteniendo distintos resultados que serán comentados a continuación. Asimismo, se ha utilizado Transfer Learning vía ResNet50 para compararla con la red neuronal propuesta.

## Primera red neuronal propuesta
Se ha propuesto una primera red neuronal basada en redes convolucionales y fully-conected. Esta, se puede encontrar en el cuaderno de Jupyter llamado "Proyecto_Convolusional_Tumor_Cerebral".
Con esta red neuronal se ha obtenido los siguientes resultados, tras entrenarla con un total de 10 epochs (estos resultados pueden ser mejorados con más entrenamiento, pero como prueba de conceptos es suficiente lo abordado en este trabajo).

Importamos Pytorch


```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
```

Transformamos las imágenes 


```python
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Redimensiona todas las imágenes a 512x512
    transforms.ToTensor(),  # Convertimos imágenes a Tensores
    transforms.Normalize((0.5,), (0.5,))  # Normalizamos con media 0.5 y desviación estándar 0.5
])
```

Obtenemos los datos de tumor cerebral (hay tres categorías de tumor y otra categoría de no tumor). Las imágenes son de 512x512 píxeles


```python
# Cargamos el dataset
path_train = r"C:\Users\34620\Desktop\ULPGC\Master\Primer semestre\Computacion Inteligente\TumorCerebralDatabase\Training" 
path_test = r"C:\Users\34620\Desktop\ULPGC\Master\Primer semestre\Computacion Inteligente\TumorCerebralDatabase\Testing"
train_data = datasets.ImageFolder(root=path_train, transform=transform)
test_data = datasets.ImageFolder(root=path_test, transform=transform)

# DataLoader para los conjuntos de entrenamiento y prueba
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
```

Redes neuronales


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Tamaño de la imagen de entrada: 3x512x512

        # Capas convolucionales
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)   # Entrada: 3 canales, Salida: 32 canales
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)        # Max pooling con un kernel de 2x2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)  # Entrada: 32 canales, Salida: 64 canales
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # Entrada: 64 canales, Salida: 128 canales
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1) # Entrada: 128 canales, Salida: 256 canales
        
        # Capas fully connected
        self.fc1 = nn.Linear(256 * 32 * 32, 512) # Aplanar a 32x32 para la primera capa fully connected
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 4) # 4 clases de salida         

    def forward(self, x): 
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        # Aplanamos para la capa lineal
        x = x.view(-1, 256 * 32 * 32)  
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Capa de salida
        
        return x

```


```python
import torch.optim as optim

model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```


```python

# Entrenamiento
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()  # Limpiamos los gradientes
        outputs = model(images)  # Pasamos las imágenes por la red
        loss = criterion(outputs, labels)  # Calculamos la pérdida
        loss.backward()  # Backpropagation
        optimizer.step()  # Actualizamos los pesos
        
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
```

    Epoch 1, Loss: 0.717665958073404
    Epoch 2, Loss: 0.3360781949427393
    Epoch 3, Loss: 0.15339633321596516
    Epoch 4, Loss: 0.1013386271842238
    Epoch 5, Loss: 0.054086835586672856


Guardamos el modelo entrenado


```python
torch.save(model.state_dict(), 'modelo_entrenado.pth')
```

Modelo entrenado guardado


```python
# Crear una nueva instancia del modelo
model = ConvNet()  # Usa la misma clase y arquitectura que la original

# Cargar los parámetros del modelo entrenado
import torch.optim as optim
model.load_state_dict(torch.load('modelo_entrenado.pth'))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

    C:\Users\34620\AppData\Local\Temp\ipykernel_11660\3172320796.py:6: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      model.load_state_dict(torch.load('modelo_entrenado.pth'))



```python
correct = 0
total = 0
with torch.no_grad():  # No necesitamos calcular gradientes para la evaluación
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the 1000 test images: {100 * correct / total}%')
```

    Accuracy of the model on the 10000 test images: 91.38062547673532%


Segundo entrenamiento


```python
# Entrenamiento
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()  # Limpiamos los gradientes
        outputs = model(images)  # Pasamos las imágenes por la red
        loss = criterion(outputs, labels)  # Calculamos la pérdida
        loss.backward()  # Backpropagation
        optimizer.step()  # Actualizamos los pesos
        
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
```

    Epoch 1, Loss: 0.07131760312735827
    Epoch 2, Loss: 0.024409494996588264
    Epoch 3, Loss: 0.03871911710270474
    Epoch 4, Loss: 0.033330853973651146
    Epoch 5, Loss: 0.0191059767058808


Guardamos el modelo con el segundo entrenamiento


```python
torch.save(model.state_dict(), 'modelo_entrenado_2.pth')
```


```python
from collections import defaultdict

# Inicialización de variables
correct = 0
total = 0
contador = 0

# Inicialización de contadores por clase
class_names = ['notumor', 'meningioma', 'glioma', 'pituitary']
true_positives = defaultdict(int)  # TP para cada clase
false_positives = defaultdict(int)  # FP para cada clase
false_negatives = defaultdict(int)  # FN para cada clase
# Cambiar el modelo al modo evaluación
model.eval()

with torch.no_grad():  # No necesitamos calcular gradientes para la evaluación
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Iterar sobre cada elemento del batch para el análisis detallado
        for label, prediction in zip(labels, predicted):
            if label == prediction:
                true_positives[class_names[label.item()]] += 1
            else:
                # Falso negativo: la clase real es `label` pero fue clasificada como algo más
                false_negatives[class_names[label.item()]] += 1
                # Falso positivo: predijo `prediction` incorrectamente
                false_positives[class_names[prediction.item()]] += 1

        contador += 1

# Resultados
accuracy = 100 * correct / total
print(f'Accuracy of the model on the {contador} test batches: {accuracy:.2f}%')

# Reporte detallado
print("\nDetailed Analysis:")
for class_name in class_names:
    tp = true_positives[class_name]
    fp = false_positives[class_name]
    fn = false_negatives[class_name]
    print(f"Class: {class_name}")
    print(f"  True Positives: {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    if tp + fn > 0:
        print(f"  Sensitivity (Recall): {tp / (tp + fn):.2f}")
    if tp + fp > 0:
        print(f"  Precision: {tp / (tp + fp):.2f}")
    print()

```

    Accuracy of the model on the 21 test batches: 95.73%
    
    Detailed Analysis:
    Class: notumor
      True Positives: 270
      False Positives: 10
      False Negatives: 30
      Sensitivity (Recall): 0.90
      Precision: 0.96
    
    Class: meningioma
      True Positives: 285
      False Positives: 30
      False Negatives: 21
      Sensitivity (Recall): 0.93
      Precision: 0.90
    
    Class: glioma
      True Positives: 404
      False Positives: 11
      False Negatives: 1
      Sensitivity (Recall): 1.00
      Precision: 0.97
    
    Class: pituitary
      True Positives: 296
      False Positives: 5
      False Negatives: 4
      Sensitivity (Recall): 0.99
      Precision: 0.98


## Segunda red neuronal propuesta
Para esta red, se ha utilizado Transfer Learning donde se ha utilizado la red residual ResNet50. En ella hemos dejado todas las capas congeladas y hemos añadidos varias capas fully-connected al final para que la salida sea una clasificación entre las cuatro clases citadas.
Con esta red neuronal se ha obtenido los siguientes resultados, tras entrenarla con un total de 20 epochs (estos resultados pueden ser mejorados con más entrenamiento, pero como prueba de conceptos es suficiente lo abordado en este trabajo). Cabe mencionar, que hemos realizado el doble de epochs y aun así hemos tartado menos tiempo que con la primera red neuronal. No obstante, como se verá a continuación, los resultados no mejoran a la primera, esto puede ser debido a que como es un problema muy específico la ResNet50 no ha sido entrenada para el mismo ni para algo similar.

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
```


```python
from torchvision import transforms

# Transformaciones para redimensionar y normalizar
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Cambiar tamaño a 224x224 debido a que ResNet fue entrenado con imágenes de ese tamaño
    transforms.ToTensor(),         # Convertir a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizar como ImageNet (usada en ResNet50)
])

```


```python
# Cargamos el dataset
path_train = r"C:\Users\34620\Desktop\ULPGC\Master\Primer semestre\Computacion Inteligente\TumorCerebralDatabase\Training" 
path_test = r"C:\Users\34620\Desktop\ULPGC\Master\Primer semestre\Computacion Inteligente\TumorCerebralDatabase\Testing"
train_data = datasets.ImageFolder(root=path_train, transform=transform)
test_data = datasets.ImageFolder(root=path_test, transform=transform)

# DataLoader para los conjuntos de entrenamiento y prueba
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
```


```python
# Cargar ResNet50 preentrenado
resnet50 = models.resnet50(pretrained=True)

# Congelar las capas de la base para que no sean entrenadas
for param in resnet50.parameters():
    param.requires_grad = False
```

    c:\Users\34620\AppData\Local\Programs\Python\Python312\Lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
      warnings.warn(
    c:\Users\34620\AppData\Local\Programs\Python\Python312\Lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.
      warnings.warn(msg)



```python
# Número de clases en tu caso
num_classes = 4

# Redefinir la cabeza de la red
resnet50.fc = nn.Sequential(
    nn.Linear(2048, 512),  # Primera capa fully connected
    nn.ReLU(),
    nn.Dropout(0.3),       # Regularización
    nn.Linear(512, 128),   # Segunda capa fully connected
    nn.ReLU(),
    nn.Dropout(0.3),       # Regularización
    nn.Linear(128, num_classes)  # Última capa fully connected (salida)
)
```


```python
# Definir función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()  # Para clasificación multiclase
optimizer = torch.optim.Adam(resnet50.fc.parameters(), lr=0.001)  # Entrenar solo la nueva cabeza
```

Entrenamos 10 épocas (este código fue ejecutado dos veces, por lo que el número final de epochs fueron 20)


```python
# Mover el modelo a GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50 = resnet50.to(device)
num_epochs = 10
# Ciclo de entrenamiento (simplificado)
for epoch in range(num_epochs):
    resnet50.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward
        optimizer.zero_grad()
        outputs = resnet50(inputs)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

```

    Epoch 1/10, Loss: 0.2463457537346712
    Epoch 2/10, Loss: 0.2433420174115197
    Epoch 3/10, Loss: 0.2212900943513023
    Epoch 4/10, Loss: 0.24149440888669238
    Epoch 5/10, Loss: 0.21729355053611973
    Epoch 6/10, Loss: 0.24481187840954885
    Epoch 7/10, Loss: 0.20715328888686677
    Epoch 8/10, Loss: 0.2256240935153302
    Epoch 9/10, Loss: 0.20894938662606577
    Epoch 10/10, Loss: 0.21736109321260585


Evaluación del modelo


```python
from collections import defaultdict


# Inicialización de variables
correct = 0
total = 0
contador = 0

# Definición de clases
class_names = ['notumor', 'meningioma', 'glioma', 'pituitary']

# Inicialización de contadores por clase
true_positives = defaultdict(int)  # TP para cada clase
false_positives = defaultdict(int)  # FP para cada clase
false_negatives = defaultdict(int)  # FN para cada clase

# Cambiar el modelo al modo evaluación
resnet50.eval()

# Configuración del dispositivo (GPU o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50.to(device)

with torch.no_grad():  # No necesitamos calcular gradientes para la evaluación
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)  # Mover a dispositivo

        outputs = resnet50(images)
        _, predicted = torch.max(outputs.data, 1)  # Predicciones

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Iterar sobre cada elemento del batch para el análisis detallado
        for label, prediction in zip(labels, predicted):
            if label == prediction:
                true_positives[label.item()] += 1
            else:
                # Falso negativo: la clase real es `label` pero fue clasificada como algo más
                false_negatives[label.item()] += 1
                # Falso positivo: predijo `prediction` incorrectamente
                false_positives[prediction.item()] += 1

        contador += 1

# Resultados
accuracy = 100 * correct / total
print(f'Accuracy of the model on the {contador} test batches: {accuracy:.2f}%')

# Reporte detallado
print("\nDetailed Analysis:")
for idx, class_name in enumerate(class_names):
    tp = true_positives[idx]
    fp = false_positives[idx]
    fn = false_negatives[idx]

    print(f"Class: {class_name}")
    print(f"  True Positives: {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")

    # Cálculo de métricas con manejo de divisiones por cero
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    print(f"  Sensitivity (Recall): {sensitivity:.2f}")
    print(f"  Precision: {precision:.2f}")
    print()

```

    Accuracy of the model on the 41 test batches: 91.38%
    
    Detailed Analysis:
    Class: notumor
      True Positives: 255
      False Positives: 21
      False Negatives: 45
      Sensitivity (Recall): 0.85
      Precision: 0.92
    
    Class: meningioma
      True Positives: 246
      False Positives: 49
      False Negatives: 60
      Sensitivity (Recall): 0.80
      Precision: 0.83
    
    Class: glioma
      True Positives: 405
      False Positives: 31
      False Negatives: 0
      Sensitivity (Recall): 1.00
      Precision: 0.93
    
    Class: pituitary
      True Positives: 292
      False Positives: 12
      False Negatives: 8
      Sensitivity (Recall): 0.97
      Precision: 0.96
    



```python
torch.save(resnet50.state_dict(), 'resnet50_brain_tumor_1.pth')  # Guardar el modelo
```


```python
# Cargar el modelo
torch.load('resnet50_brain_tumor_1.pth')
```

Se puede observar que los resultados con esta red, tras realizar más epochs de entrenamiento, no superan a la propuesta, sin embargo, el tiempo de ejecución por epoch de esta último si es muy inferior al de la primera como se había comentado.

<footer>

<!--
  <<< Author notes: Footer >>>
  Add a link to get support, GitHub status page, code of conduct, license link.
-->

---

Get help: [Post in our discussion board](https://github.com/orgs/skills/discussions/categories/github-pages) &bull; [Review the GitHub status page](https://www.githubstatus.com/)

&copy; 2023 GitHub &bull; [Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/code_of_conduct.md) &bull; [MIT License](https://gh.io/mit)

</footer>
