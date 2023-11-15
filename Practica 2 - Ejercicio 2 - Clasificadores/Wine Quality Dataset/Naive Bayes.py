import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Carga el conjunto de datos desde la URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
data = pd.read_csv(url, sep=';')  # Asegúrate de que el separador sea el punto y coma (;)

# Divide los datos en características (X) y etiquetas (y)
X = data.drop('quality', axis=1)  # Excluye la columna 'quality' como característica
y = (data['quality'] > 6).astype(int)  # Convierte la calidad en una etiqueta binaria (0 o 1)

# Divide los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crea un modelo Naive Bayes Gaussiano
model = GaussianNB()

# Ajusta el modelo a los datos de entrenamiento
model.fit(X_train, y_train)

# Realiza predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcula la precisión
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Calcula la precisión (precision)
precision = precision_score(y_test, y_pred)
print(f'Precision: {precision:.2f}')

# Calcula la sensibilidad (recall)
sensitivity = recall_score(y_test, y_pred)
print(f'Sensitivity: {sensitivity:.2f}')

# Calcula la especificidad
specificity = sum((y_test == 0) & (y_pred == 0)) / sum(y_test == 0)
print(f'Specificity: {specificity:.2f}')

# Calcula el puntaje F1
f1 = f1_score(y_test, y_pred)
print(f'F1 Score: {f1:.2f}')

# Calcula la matriz de confusión
confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(confusion)
