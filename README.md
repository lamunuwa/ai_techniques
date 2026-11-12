# Tecnicas basicas de Machine Learning

El objetivo de este repositorio es darte una introducción práctica y colaborativa a los algoritmos fundamentales de Machine Learning. A través de este proyecto aprenderás a comprender la intuición detrás de cada técnica, implementarla desde cero utilizando NumPy, compararla con las herramientas estándar de scikit-learn y trabajar en equipo adoptando buenas prácticas de desarrollo.

---

## 1. Técnicas cubiertas

| # | Carpeta | Técnica | Para qué sirve |
|---|---------|---------|----------------------------|
| 1 | `regres/` | Regresión lineal | Predecir valores continuos y estimar probabilidades. |
| 2 | `knn/` | K-Nearest Neighbors | Clasificar o estimar valores basándose en la proximidad con las muestras más cercanas. |
| 3 | `arbol/` | Árbol de decisión | Particionar datos mediante reglas condicionales jerárquicas para clasificación y regresión. |
| 4 | `rforest/` | Random Forest | Mejorar precisión y reducir varianza combinando múltiples árboles en paralelo. |
| 5 | `gboost/` | Gradient Boosting | Reducir errores de predicción entrenando árboles de forma secuencial y correctiva (boosting). |
| 6 | `iforest/` | Isolation Forest | Identificar observaciones anómalas o atípicas aislando instancias mediante particiones aleatorias. |

Orden recomendado: cada técnica reutiliza ideas de la anterior (un bosque son muchos árboles, boosting son árboles encadenados).

---

## 2. Estructura del repositorio

```
├── README.md                  # Este archivo: plan y convenciones
├── FUNDAMENTOS.md             # Conceptos generales de ML
├── requirements.txt           # Dependencias con versiones fijas
├── pyproject.toml             # Metadatos del proyecto y configuración de herramientas
├── .gitignore
├── .github/
│   └── pull_request_template.md
 ├── data/                      # Cargado de datasets privados
├── regresiones/
│   ├── README.md              # Qué es, idea, restricciones, cuándo usarlo y cuándo no
│   ├── main.py                # Implementación conjunta (NumPy + scikit-learn)
│   ├── integrante1/           # Notebooks personales
│   ├── integrante2/
│   └── integrante3/
├── knn/                       # Misma estructura
├── arbol/
├── rforest/
├── gboost/
└── iforest/
```

Notas de diseño:
- Una carpeta por participante dentro de cada modelo. Evita conflictos de merge en los notebooks, cada quien solo edita los suyos.
- El dataset es unico y propio, asi usas el que quieras y evitamos subir archivos enormes al repositorio.

---

## 3. Cómo se trabaja cada modelo

Cada técnica sigue el mismo ciclo:

1. Leer el `README.md` del modelo (qué es, idea, restricciones).
2. En tu carpeta explora el modelo paso a paso.
   - Puedes empezar implementando con Numpy, un modelo simple.
   - Compárala con Scikit-Learn sobre el mismo dataset y explica las diferencias.
3. Entre todas armamos un solo script con NumPy y scikit-learn, dentro de `main.py`. Cada participante puede aportar al menos una parte (preparación de datos, modelo desde cero, modelo con sklearn, evaluación).
4. Una vez terminado puedes agregar algo de información relevante al README. 
5. Cada aportación entra por pull request y debe ser revisado por otro integrante.
6. Cierre del modelo

## 4. Convenciones del equipo

### 4.1 Ramas

- `main` está protegida: no se hace push directo; todo entra por PR.
- Formato: `tipo/alcance-descripcion-corta`, en minúsculas y con guiones.
- Ejemplos: `feat/knn-main-evaluacion`, `docs/fundamentos-ml`, `fix/common-split-estratificado`.

### 4.2 Commits

Formato: `tipo(alcance): descripción en imperativo`, máximo ~72 caracteres, en español. Siguiendo [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

Mejor varios commits pequeños que uno gigante.

### 4.3 Issues

- Una issue por tarea concreta. Título: `[modelo] tarea` (por ejemplo `[knn] escribir README del modelo`).
- Descripción con:
  - Qué hay que hacer.
  - Criterio de hecho y responsable asignada.
- Puedes usar labels, projects y todas las herramientas de los Issues.

### 4.4 Pull requests

- Título con el mismo formato de commit: `feat(knn): ...`.
- La descripción usa la plantilla.
- Para hacer merge, debe haber una revisión.
- Enlazar la issue con `Closes #N`.
- Merge y borrar la rama después.

---

## 5. Configuración del entorno

Puedes instalar todo en local, pero es preferible usar un .venv para no ensuciar tu sistema.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name ml-3b
pre-commit install
```
