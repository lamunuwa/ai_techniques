# Random Forest (Bosques Aleatorios)

Espacio de trabajo conjunto para el estudio e implementación del algoritmo Random Forest.

---

## 1. Qué es
Un método de ensamble (*ensemble learning*) supervisado basado en **Bagging** (*Bootstrap Aggregating*) que combina predicciones de múltiples árboles de decisión individuales para mejorar la precisión y robustez.

---

## 2. Idea principal
Reducir la alta varianza inherente a un árbol individual promediando las salidas de un conjunto de árboles descorrelacionados entrenados en paralelo:
1. **Bootstrap:** Cada árbol se entrena con un subconjunto aleatorio de muestras con reemplazo.
2. **Selección aleatoria de variables (*Feature Bagging*):** En cada división o nodo, el árbol solo evalúa un subconjunto aleatorio de características (típicamente $\sqrt{d}$ para clasificación).
3. **Agregación:** En clasificación se toma el voto mayoritario; en regresión, el promedio de todos los árboles.

---

## 3. Restricciones
- **Pérdida de interpretabilidad:** Se pierde la simplicidad visual de un único árbol, convirtiéndose en un modelo de mayor complejidad (*black-box* relativo).
- **Costo computacional y memoria:** Entrenar y almacenar decenas o cientos de árboles demanda más tiempo y recursos.
- **Sin extrapolación en regresión:** Al basarse en árboles, no puede predecir valores fuera del rango observado durante el entrenamiento.

---

## 4. Cuándo usarlo y cuándo no
- **Cuándo usarlo:**
  - En datos tabulares complejos donde se busque un rendimiento sobresaliente sin requerir un ajuste minucioso de hiperparámetros.
  - Como punto de referencia de alta calidad frente a un árbol individual o modelos lineales.
  - Cuando se requiera medir la importancia relativa de variables (*feature importances*).
- **Cuándo no usarlo:**
  - En entornos con restricciones severas de memoria o que exijan inferencias de latencia extremadamente baja.
  - Cuando la explicabilidad explícita regla por regla sea un requisito estricto de negocio o auditoría.
