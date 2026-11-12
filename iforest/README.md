# Isolation Forest

Espacio de trabajo conjunto para el estudio e implementación del algoritmo Isolation Forest para detección de anomalías.

---

## 1. Qué es
Un algoritmo de aprendizaje **no supervisado** diseñado específicamente para identificar observaciones anómalas o valores atípicos (*outliers*) en un conjunto de datos.

---

## 2. Idea principal
Aislar instancias individuales mediante particiones aleatorias en lugar de modelar la distribución de puntos normales:
- Se seleccionan variables al azar y cortes uniformes entre sus valores mínimo y máximo.
- Dado que las anomalías son "pocas y diferentes", resultan mucho más fáciles de aislar y quedan confinadas en hojas con **caminos muy cortos** desde la raíz del árbol.
- Las muestras normales se encuentran agrupadas densamente y requieren un mayor número de divisiones (caminos más largos).

---

## 3. Restricciones
- **Naturaleza no supervisada:** No optimiza directamente respecto a una etiqueta de verdad terreno (a menos que se disponga de un conjunto de prueba etiquetado para calibración posterior).
- **Parámetro de contaminación:** Requiere estimar o conocer aproximadamente la proporción de anomalías presentes en el dataset (`contamination`).
- **Particiones ortogonales:** Al igual que los árboles convencionales, los cortes rectos pueden generar artefactos espaciales si existen dependencias complejas entre variables.

---

## 4. Cuándo usarlo y cuándo no
- **Cuándo usarlo:**
  - Detección de fraudes, fallos de maquinaria, intrusiones en sistemas o limpieza de anomalías en datasets tabulares.
  - Datasets de volumen moderado a grande donde los métodos basados en distancias (como KNN) resultan ineficientes.
- **Cuándo no usarlo:**
  - Cuando se dispone de un conjunto balanceado o con suficientes etiquetas de anomalías donde la clasificación supervisada rinda mejor.
  - En datos con un volumen masivo de características irrelevantes que diluyan la efectividad de las particiones aleatorias.
