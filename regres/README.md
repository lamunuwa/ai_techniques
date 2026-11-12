# Regresiones (Lineal y Logística)

Espacio de trabajo conjunto para el estudio e implementación de modelos de regresión lineal y logística.

---

## 1. Qué es
Son modelos supervisados fundamentales que modelan la relación entre una o más variables independientes ($X$) y una variable objetivo ($y$):
- **Regresión Lineal:** Predice una variable objetivo continua.
- **Regresión Logística:** Modela la probabilidad de pertenencia a una clase discreta (clasificación) mediante la función sigmoide.

---

## 2. Idea principal
Aprender un vector de coeficientes o pesos $\beta$ y un término independiente $\beta_0$ que combine linealmente las variables de entrada:
$$\hat{y} = \beta_0 + \beta_1 x_1 + \dots + \beta_d x_d$$
- En regresión lineal se minimiza la suma de errores al cuadrado (MSE/MCO).
- En regresión logística se maximiza la verosimilitud (minimización de la función de pérdida logarítmica o Binary Cross-Entropy).

---

## 3. Restricciones
- **Linealidad:** Asume que la relación entre variables y el objetivo (o el logit) es lineal.
- **Sensibilidad a outliers:** Los valores atípicos pueden desviar fuertemente la recta o frontera aprendida.
- **Multicolinealidad:** Variables altamente correlacionadas pueden generar inestabilidad en la estimación de los coeficientes.
- **Escalado:** Indispensable cuando se aplican técnicas de regularización ($L_1$/Lasso, $L_2$/Ridge).

---

## 4. Cuándo usarlo y cuándo no
- **Cuándo usarlo:**
  - Como primera línea base (baseline) rápida, interpretable y económica computacionalmente.
  - Cuando se necesita explicar el impacto directo y cuantificable de cada variable sobre el resultado.
  - En problemas donde las relaciones son fundamentalmente lineales.
- **Cuándo no usarlo:**
  - Si existen dependencias altamente no lineales o complejas interacciones entre variables que no se hayan transformado manualmente.
  - Cuando la interpretabilidad no es prioridad y modelos de mayor capacidad superan holgadamente su desempeño.
