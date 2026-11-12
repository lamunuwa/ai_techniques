# Gradient Boosting

Espacio de trabajo conjunto para el estudio e implementación del algoritmo Gradient Boosting.

---

## 1. Qué es
Un método de ensamble (*ensemble learning*) supervisado basado en **Boosting** que entrena modelos base (usualmente árboles de decisión simples o *stumps*) de manera secuencial y correctiva.

---

## 2. Idea principal
Optimizar de forma iterativa una función de pérdida arbitraria agregando modelos base que apunten en la dirección del gradiente negativo (los residuos de predicción):
$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$
- Cada nuevo árbol aprende a corregir los errores cometidos por los árboles precedentes.
- El parámetro de tasa de aprendizaje (*learning rate*, $\eta$) modula el aporte de cada nuevo estimador para evitar sobreajuste acelerado.

---

## 3. Restricciones
- **Riesgo de sobreajuste (overfitting):** Si se añaden demasiados estimadores sin regularización o con un *learning rate* alto, el modelo terminará memorizando el ruido.
- **Sensibilidad a hiperparámetros:** Requiere un ajuste cuidadoso de `learning_rate`, `n_estimators`, `max_depth` y parámetros de submuestreo.
- **Entrenamiento secuencial:** Dado que cada paso depende de los errores del anterior, el entrenamiento básico no es paralelizable de la misma forma que Bagging.

---

## 4. Cuándo usarlo y cuándo no
- **Cuándo usarlo:**
  - Cuando el objetivo principal sea exprimir al máximo la precisión en datos tabulares estructurados (el estándar de la industria y competencias).
  - Cuando se cuente con tiempo y recursos para optimizar hiperparámetros.
- **Cuándo no usarlo:**
  - En datos con niveles extremos de ruido donde el boosting tienda a sobreajustar las muestras anómalas.
  - Si se busca un modelo rápido de entrenar sin necesidad de calibración fina.
