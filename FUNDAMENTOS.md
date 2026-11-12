# Fundamentos Generales de Machine Learning

Este documento reúne los conceptos teóricos y metodológicos esenciales que aplican a cualquier proyecto de Machine Learning, con independencia del algoritmo específico utilizado.

---

## 1. Aprendizaje Supervisado vs. No Supervisado

### Aprendizaje Supervisado
Se dispone de un conjunto de datos etiquetado $\{(x_i, y_i)\}_{i=1}^N$, donde cada muestra de entrada $x_i \in \mathbb{R}^d$ cuenta con una etiqueta o valor objetivo $y_i$ asociado. El objetivo es aprender una función $f: X \rightarrow Y$ que aproxime las relaciones subyacentes y permita predecir $y$ para nuevas entradas $x$.

Se divide principalmente en dos tareas:
- **Regresión:** La variable objetivo $y$ es continua ($y \in \mathbb{R}$). Ejemplos: predecir el precio de una vivienda, la temperatura o la demanda energética.
- **Clasificación:** La variable objetivo $y$ es discreta o categórica ($y \in \{C_1, C_2, \dots, C_k\}$). Puede ser binaria (ej. detección de spam: 0 o 1) o multiclase (ej. reconocimiento de dígitos: 0 a 9).

### Aprendizaje No Supervisado
Solo se dispone de entradas $\{x_i\}_{i=1}^N$ sin etiquetas asociadas ($y$). El objetivo consiste en descubrir estructuras, patrones, agrupamientos o distribuciones ocultas en los datos.

Principales enfoques:
- **Clustering:** Agrupación de observaciones similares (ej. K-Means, DBSCAN).
- **Reducción de dimensionalidad:** Proyección de datos a espacios de menor dimensión preservando varianza o distancias (ej. PCA, t-SNE, UMAP).
- **Detección de anomalías:** Identificación de patrones atípicos o instancias que difieren del comportamiento normal (ej. Isolation Forest, One-Class SVM).

---

## 2. Datos de Entrenamiento, Validación y Prueba

Para evaluar de forma realista la capacidad de un modelo, el conjunto de datos completo debe dividirse en subconjuntos disjuntos:

```
[========================== Dataset Completo ==========================]
[------- Train (Entrenamiento) -------] [--- Val (Validación) ---] [--- Test (Prueba) ---]
                70%                               15%                       15%
```

- **Entrenamiento (Train):** Conjunto utilizado por el algoritmo de optimización para ajustar los parámetros internos del modelo (pesos, coeficientes, umbrales).
- **Validación (Validation):** Conjunto empleado para tomar decisiones de diseño: selección de modelos, ajuste de hiperparámetros y detección temprana de sobreajuste (early stopping). No participa directamente en la actualización de los pesos.
- **Prueba (Test):** Conjunto reservado exclusivamente para la evaluación final no sesgada del rendimiento del modelo seleccionado. **Nunca** debe utilizarse para tomar decisiones de modelado ni ajuste de parámetros.

### Proporciones habituales
- **Con split triple (Train / Val / Test):** 70% / 15% / 15% o 60% / 20% / 20%.
- **Con validación cruzada (Train / Test):** 80% / 20% o 75% / 25%, aplicando validación cruzada (K-Fold) dentro del 80% de entrenamiento.

---

## 3. Generalización

La **generalización** es la capacidad de un modelo matemático para predecir con precisión sobre datos nuevos, no observados durante su entrenamiento, provenientes de la misma distribución generadora de datos.

Un modelo que memoriza los datos de entrenamiento pero fracasa frente a nuevas observaciones carece de valor práctico. Por ende, la métrica que determina la utilidad real de una solución en producción es siempre el rendimiento sobre el conjunto de prueba o validación, nunca sobre el conjunto de entrenamiento.

---

## 4. Sobreajuste (Overfitting) y Subajuste (Underfitting)

El ajuste del modelo describe cómo se adapta su capacidad a la complejidad del problema:

- **Subajuste (Underfitting):** Ocurre cuando el modelo es demasiado simple para capturar la estructura de los datos (alta tasa de error tanto en entrenamiento como en validación).
  - *Causas:* Modelo con poca capacidad representativa (ej. modelo lineal para datos altamente no lineales), regularización excesiva o falta de características relevantes.
  - *Solución:* Aumentar la complejidad del modelo, reducir regularización, ingeniería de variables.
- **Sobreajuste (Overfitting):** Ocurre cuando el modelo memoriza el ruido y las particularidades de la muestra de entrenamiento en lugar de aprender el patrón subyacente (bajo error en entrenamiento, pero alto error en validación/prueba).
  - *Causas:* Modelo excesivamente complejo para la cantidad de datos disponibles, entrenamiento durante demasiadas épocas o falta de regularización.
  - *Solución:* Regularización (L1/L2, dropout), poda de árboles, recolección de más datos o reducción de dimensionalidad.

### Detección mediante curvas de aprendizaje
Al graficar el error (o métrica) frente a la complejidad del modelo o las épocas de entrenamiento:
- Si el error de validación comienza a divergir del error de entrenamiento (el de train baja y el de val sube), se ha entrado en la zona de **sobreajuste**.

---

## 5. Compromiso Sesgo-Varianza (Bias-Variance Tradeoff)

El error de generalización esperado de cualquier modelo se puede descomponer matemáticamente en tres componentes:

$$\text{Error Total} = \text{Sesgo}^2 + \text{Varianza} + \text{Ruido Irreducible}$$

- **Sesgo (Bias):** Error introducido por aproximar un problema del mundo real mediante un modelo simplificado. Un sesgo alto genera **subajuste**.
- **Varianza (Variance):** Sensibilidad del modelo a fluctuaciones en el conjunto de entrenamiento. Una varianza alta genera **sobreajuste** (pequeños cambios en los datos de entrada producen predicciones drásticamente distintas).
- **Ruido Irreducible:** Límite inferior del error causado por factores no medidos o aleatoriedad intrínseca del proceso.

El objetivo del diseño de modelos es encontrar el punto óptimo de complejidad que minimice la suma de sesgo al cuadrado y varianza.

---

## 6. Validación Cruzada (K-Fold Cross-Validation)

En lugar de un split simple de validación, la validación cruzada divide los datos de entrenamiento en $K$ particiones (folds) de tamaño similar:

```
Fold 1        Fold 2        Fold 3        Fold 4        Fold 5
[ Test  ]    [ Train ]    [ Train ]    [ Train ]    [ Train ]  -> Métrica 1
[ Train ]    [ Test  ]    [ Train ]    [ Train ]    [ Train ]  -> Métrica 2
[ Train ]    [ Train ]    [ Test  ]    [ Train ]    [ Train ]  -> Métrica 3
[ Train ]    [ Train ]    [ Train ]    [ Test  ]    [ Train ]  -> Métrica 4
[ Train ]    [ Train ]    [ Train ]    [ Train ]    [ Test  ]  -> Métrica 5

Rendimiento Final = Promedio(Métricas 1..5) ± Desviación Estándar
```

### Variantes clave:
- **K-Fold estándar:** Útil para problemas de regresión o clases perfectamente balanceadas. Típicamente $K=5$ o $K=10$.
- **Stratified K-Fold:** Para clasificación; asegura que cada fold mantenga la misma proporción porcentual de clases que el conjunto original.
- **Time Series Split:** Para datos temporales; el entrenamiento solo incluye datos pasados para predecir folds futuros, evitando mirar hacia adelante en el tiempo.

---

## 7. Fuga de Datos (Data Leakage)

La **fuga de datos** ocurre cuando información ajena al conjunto de entrenamiento (proveniente del futuro o del conjunto de prueba/validación) se introduce inadvertidamente durante el proceso de aprendizaje. 

Es uno de los errores metodológicos más costosos, ya que genera estimaciones de rendimiento sumamente optimistas durante el desarrollo que colapsan en producción.

### Ejemplo clásico: Normalización antes de separar datos
- **Error:** Calcular la media y desviación estándar de todo el dataset completo y luego dividir en train y test. El conjunto de prueba ya influyó en el escalado de entrenamiento.
- **Forma correcta:**
  1. Dividir primero en `X_train` y `X_test`.
  2. Ajustar el transformador (`scaler.fit(X_train)`).
  3. Aplicar la transformación a ambos (`scaler.transform(X_train)`, `scaler.transform(X_test)`).
- **Mejor práctica:** Usar pipelines (ej. `Pipeline` de scikit-learn) para encapsular transformaciones y estimadores.

---

## 8. Métricas de Evaluación

### Métricas de Regresión
- **MAE (Mean Absolute Error):** $\frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$. Mide el error promedio en las mismas unidades de la variable objetivo. Es robusto frente a valores atípicos.
- **RMSE (Root Mean Squared Error):** $\sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$. Penaliza con mayor severidad los errores grandes debido al término cuadrático.
- **$R^2$ (Coeficiente de determinación):** Proporción de la varianza total de $y$ explicada por el modelo ($1 - \frac{SS_{res}}{SS_{tot}}$). Un valor de 1.0 indica predicción perfecta; 0.0 indica que rinde igual que predecir la media constante.

### Métricas de Clasificación
- **Matriz de Confusión:**
  - $TP$ (Verdaderos Positivos), $FP$ (Falsos Positivos)
  - $FN$ (Falsos Negativos), $TN$ (Verdaderos Negativos)
- **Accuracy (Exactitud):** $\frac{TP + TN}{TP + TN + FP + FN}$. Proporción de aciertos globales.
  - *Advertencia con clases desbalanceadas:* Si el 99% de los casos son negativos (ej. detección de fraudes), predecir siempre negativo otorga un 99% de accuracy, resultando en un modelo completamente inútil.
- **Precisión:** $\frac{TP}{TP + FP}$. De los que el modelo predijo positivos, ¿cuántos eran realmente positivos? (Importante cuando el costo de un falso positivo es alto).
- **Recall (Sensibilidad):** $\frac{TP}{TP + FN}$. De todos los casos positivos reales, ¿cuántos logró capturar el modelo? (Crítico cuando no se pueden tolerar falsos negativos, ej. diagnósticos médicos).
- **F1-Score:** Media armónica entre precisión y recall: $2 \cdot \frac{\text{Precisión} \cdot \text{Recall}}{\text{Precisión} + \text{Recall}}$.
- **ROC-AUC (Área bajo la curva ROC):** Mide la capacidad del modelo para discriminar entre clases a través de diferentes umbrales de decisión. Un valor de 0.5 equivale a una predicción aleatoria y 1.0 a una discriminación perfecta.

---

## 9. Preparación de Datos (Preprocesamiento)

Los datos del mundo real rara vez se encuentran listos para alimentar directamente un algoritmo:

### 1. Manejo de valores faltantes
- **Eliminación:** Borrado de filas o columnas completas (viable si la pérdida de datos es mínima o no informativa).
- **Imputación:** Relleno de nulos con estadísticas simples (media, mediana, moda) o modelos avanzados (KNNImputer, IterativeImputer).

### 2. Escalado numérico
Muchos algoritmos son sensibles a la escala de las variables (especialmente los basados en distancias como KNN o en gradientes como regresión y redes neuronales):
- **Estandarización (StandardScaler):** Transforma la variable para que tenga $\mu = 0$ y $\sigma = 1$. Ideal cuando los datos siguen una distribución aproximadamente normal.
- **Normalización Min-Max (MinMaxScaler):** Escala los valores a un rango fijo, típicamente $[0, 1]$. Sensible a valores atípicos.
- *Nota:* Los árboles de decisión y ensambles basados en árboles no se ven afectados por escalas monótonas de variables continuas.

### 3. Codificación de variables categóricas
- **One-Hot Encoding:** Crea una columna binaria por cada categoría única. Adecuado para variables nominales sin orden intrínseco (ej. países, colores). Cuidado con la alta dimensionalidad.
- **Ordinal Encoding:** Mapea categorías a números enteros respetando un orden intrínseco (ej. bajo = 1, medio = 2, alto = 3).

---

## 10. Hiperparámetros vs. Parámetros

- **Parámetros:** Valores que el modelo aprende automáticamente a partir de los datos durante el entrenamiento (ej. coeficientes $\beta$ en regresión, pesos en redes neuronales, umbrales de división en árboles).
- **Hiperparámetros:** Configuraciones externas definidas por quien diseña el modelo antes de iniciar el entrenamiento, que controlan el comportamiento del algoritmo de aprendizaje (ej. factor de regularización $\alpha$, profundidad máxima `max_depth`, número de vecinos `k`).

### Métodos de búsqueda de hiperparámetros
- **Grid Search:** Evalúa de manera exhaustiva todas las combinaciones posibles en una cuadrícula predefinida de valores. Garantiza encontrar el mejor punto de la rejilla, pero es computacionalmente costoso.
- **Random Search:** Muestrea combinaciones aleatorias de hiperparámetros a partir de distribuciones especificadas. Suele encontrar soluciones comparables o mejores que Grid Search en una fracción del tiempo.

---

## 11. Reproducibilidad

Garantizar que cualquier integrante del equipo pueda replicar exactamente los mismos resultados numéricos es fundamental:

1. **Semillas aleatorias fijas:** Establecer valores explícitos para los generadores pseudoaleatorios (`random_state=42`, `np.random.default_rng(42)`, etc.) en particiones de datos, inicializaciones y modelos estocásticos.
2. **Entornos aislados:** Emplear entornos virtuales (`venv`, `conda`) con versiones exactas especificadas en archivos como `requirements.txt` o `pyproject.toml`.
3. **Flujos deterministas:** Evitar dependencias del orden no controlado de iteración (como conjuntos o diccionarios sin ordenar) en pasos críticos de procesamiento.

---

## 12. Línea Base (Baseline)

Antes de implementar o afinar modelos complejos, es imperativo establecer un **modelo de línea base (baseline)** sencillo:

- **En regresión:** Predecir siempre el promedio o la mediana de la variable objetivo.
- **En clasificación:** Predecir siempre la clase mayoritaria (Dummy Classifier) o basarse en una regla heurística simple de negocio.

### ¿Por qué es crucial?
Un modelo sofisticado (como un ensamble de árboles o una red neuronal) que obtenga un MAE de 50 o un accuracy del 85% solo tiene verdadero mérito si supera significativamente a la línea base trivial (que podría obtener 55 de MAE o 82% de accuracy sin complejidad ni costo computacional). El baseline cuantifica el valor real añadido por el Machine Learning.
