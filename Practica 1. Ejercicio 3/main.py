import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos
data = pd.read_csv("concentlite.csv")

# Dividir los datos en características (X) y etiquetas (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar las características para acelerar el entrenamiento
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir la arquitectura de la red neuronal
input_size = X_train.shape[1]
hidden_layers = [4, 4]  # Cambia esta lista para configurar las capas ocultas
output_size = 1

# Función de activación (sigmoide)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función de activación (sigmoide)
def sigmoid_derivative(x):
    return x * (1 - x)

# Inicialización de pesos y sesgos
np.random.seed(0)
weights = []
biases = []

layers = [input_size] + hidden_layers + [output_size]
for i in range(1, len(layers)):
    w = np.random.uniform(-1, 1, (layers[i-1], layers[i]))
    b = np.zeros((1, layers[i]))
    weights.append(w)
    biases.append(b)

# Hiperparámetros
learning_rate = 0.1
epochs = 10000

# Entrenamiento de la red neuronal
for epoch in range(epochs):
    # Feedforward
    layer_outputs = []
    layer_inputs = [X_train]
    
    for i in range(len(hidden_layers) + 1):
        z = np.dot(layer_inputs[i], weights[i]) + biases[i]
        a = sigmoid(z)
        layer_outputs.append(a)
        layer_inputs.append(a)

    # Calcular el error
    error = y_train.reshape(-1, 1) - layer_outputs[-1]

    # Retropropagación
    deltas = [error * sigmoid_derivative(layer_outputs[-1])]
    for i in range(len(hidden_layers), 0, -1):
        delta = deltas[-1].dot(weights[i].T) * sigmoid_derivative(layer_outputs[i])
        deltas.append(delta)
    deltas = deltas[::-1]

    # Actualizar pesos y sesgos
    for i in range(len(weights)):
        weights[i] += learning_rate * layer_inputs[i].T.dot(deltas[i])
        biases[i] += learning_rate * np.sum(deltas[i], axis=0)

# Predicciones en el conjunto de prueba
layer_inputs = [X_test]
for i in range(len(hidden_layers) + 1):
    z = np.dot(layer_inputs[i], weights[i]) + biases[i]
    a = sigmoid(z)
    layer_inputs.append(a)

y_pred = layer_inputs[-1]
y_pred_class = np.round(y_pred)

# Calcular la precisión en el conjunto de prueba
accuracy = accuracy_score(y_test, y_pred_class)
print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")

# Representación gráfica de la clasificación
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], label='Clase 0', c='b', marker='o')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], label='Clase 1', c='r', marker='s')
plt.title("Clasificación por el Perceptrón Multicapa")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.legend()
plt.show()
