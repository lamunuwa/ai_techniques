# Árbol de Decisión

Espacio de trabajo conjunto para el estudio e implementación de Árboles de Decisión para clasificación y regresión.

---

## 1. Qué es
Un modelo supervisado no paramétrico que divide el espacio de datos en subregiones jerárquicas mediante una serie de reglas condicionales (preguntas binarias del tipo $x_j \le \text{umbral}$).

---

## 2. Idea principal
Construir una estructura jerárquica en forma de árbol comenzando desde un nodo raíz y dividiendo de forma recursiva (algoritmo voraz o *greedy*):
- En cada división se selecciona la variable y el umbral que maximicen la pureza del subconjunto.
- **Clasificación:** Se evalúa la reducción de impureza mediante criterios como **Gini** o **Entropía**.
- **Regresión:** Se busca minimizar la varianza de los valores observados (reducción de MSE).

---

## 3. Restricciones
- **Alta propensión al sobreajuste (overfitting):** Si se permite crecer sin restricciones, memoriza el conjunto de entrenamiento creando hojas con una sola muestra. Requiere poda (*pruning*) o control de hiperparámetros (`max_depth`, `min_samples_split`).
- **Inestabilidad (alta varianza):** Pequeñas variaciones en los datos de entrenamiento pueden provocar divisiones completamente diferentes.
- **Divisiones ortogonales:** Solo genera particiones paralelas a los ejes de coordenadas, teniendo dificultades para modelar relaciones diagonales simples.

---

## 4. Cuándo usarlo y cuándo no
- **Cuándo usarlo:**
  - Cuando se necesita interpretabilidad total y explicar visualmente las decisiones a personas sin perfil técnico.
  - Como bloque constructor base para ensambles más potentes (Random Forest, Gradient Boosting).
  - No requiere escalado ni normalización de variables numéricas.
- **Cuándo no usarlo:**
  - Si el objetivo prioritario es obtener la máxima capacidad predictiva (los ensambles suelen ser superiores).
  - En problemas con relaciones continuas complejas o datos con mucho ruido donde la inestabilidad afecta los resultados.
