# K-Nearest Neighbors (KNN)

Espacio de trabajo conjunto para el estudio e implementación del algoritmo K-Nearest Neighbors.

---

## 1. Qué es
KNN es un algoritmo supervisado no paramétrico y basado en instancias (*lazy learning*), aplicable tanto a clasificación como a regresión. No asume una distribución funcional explícita sobre los datos.

---

## 2. Idea principal
Clasificar o estimar el valor de una nueva observación en función de las $k$ muestras más próximas en el espacio de características:
- Para **clasificación**: Se toma el voto mayoritario de las clases de los $k$ vecinos.
- Para **regresión**: Se toma el promedio (o promedio ponderado por distancia) de los valores objetivo de los $k$ vecinos.
- El cálculo de cercanía se realiza mediante funciones de distancia (ej. Euclídea, Manhattan, Minkowski).

---

## 3. Restricciones
- **Escalado estricto:** Es extremadamente sensible a las escalas de las variables; normalizar o estandarizar es obligatorio.
- **Maldición de la dimensionalidad:** Su efectividad cae drásticamente en espacios con muchas dimensiones ($d \gg 1$), donde todas las distancias tienden a volverse similares.
- **Costo en inferencia:** La fase de entrenamiento es trivial (solo memoriza datos), pero la predicción requiere calcular distancias contra todo el conjunto ($O(N \cdot d)$).
- **Sensibilidad al ruido y outliers:** Un valor de $k$ pequeño es propenso a sobreajuste por ruido; un $k$ grande suaviza en exceso.

---

## 4. Cuándo usarlo y cuándo no
- **Cuándo usarlo:**
  - Datasets de tamaño moderado con pocas dimensiones ($d < 20-30$).
  - Problemas con límites de decisión no lineales donde vecinos cercanos compartan comportamientos homogéneos.
  - Como algoritmo simple y directo de recomendación o similitud.
- **Cuándo no usarlo:**
  - Datasets muy grandes o aplicaciones que requieran predicciones ultrarrápidas en tiempo real.
  - Conjuntos de datos con alta dimensionalidad (texto, imágenes sin comprimir).
