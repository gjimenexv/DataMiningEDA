# pckModelosAvanzados — Minería de Datos Avanzada en Python

Paquete modular con una **clase por cada modelo visto en el curso** BCE0019 Minería de
Datos Avanzada (LEAD University). Todos los modelos comparten el mismo flujo de uso,
las mismas métricas y los mismos gráficos, de modo que cambiar de algoritmo es cambiar
una línea. Se apoya en `pckEDA`: cada modelo **hereda de `AnalisisDatosExploratorio`**, así que puede
limpiar y explorar los datos con los mismos métodos antes de modelar.

El temario y la teoría de cada semana están en
[`Avanzada/Indice_Modelos_Mineria_Avanzada.md`](../Avanzada/Indice_Modelos_Mineria_Avanzada.md).

---

## ¿Qué incluye?

| Semana | Clase | Modelo | Clasificación | Documentación |
|:---:|---|---|---|---|
| 1 | `KNN` | K-Nearest Neighbors | Supervisado · clasificación | [KNN.md](knn/KNN.md) |
| 2 | `ArbolDecision` | Árbol de decisión (CART) | Supervisado · clasificación | [ArbolDecision.md](arbol_decision/ArbolDecision.md) |
| 3 | `Bagging` | Bagging de árboles | Supervisado · clasificación | [Bagging.md](bagging/Bagging.md) |
| 3 | `RandomForest` | Random Forest | Supervisado · clasificación | [RandomForest.md](random_forest/RandomForest.md) |
| 4 | `AdaBoost` | AdaBoost | Supervisado · clasificación | [AdaBoost.md](adaboost/AdaBoost.md) |
| 5 | `GradientBoosting` | Gradient Boosting | Supervisado · clasificación | [GradientBoosting.md](gradient_boosting/GradientBoosting.md) |
| 5 | `XGBoost` | XGBoost | Supervisado · clasificación | [XGBoost.md](xgb/XGBoost.md) |
| 6 | `LDA` | Análisis Discriminante Lineal | Supervisado · clasificación | [LDA.md](lda/LDA.md) |
| 6 | `QDA` | Análisis Discriminante Cuadrático | Supervisado · clasificación | [QDA.md](qda/QDA.md) |

> Todos los modelos vistos hasta la Semana 6 son de **aprendizaje supervisado** (usan la
> variable objetivo). Cada clase lo expone en `tipo_aprendizaje` y `tarea`, y cada `.md` lo
> indica con una etiqueta de clasificación.

Utilidades transversales: `ModeloBase` (clase madre) y `comparar_modelos` (tabla comparativa).
La validación cruzada y el tuneo de hiperparámetros (Semanas 3 y 4) viven en `ModeloBase`
y están disponibles en todos los modelos.

---

## Instalación

El paquete se usa directamente desde el directorio del proyecto:

```python
import sys
sys.path.insert(0, '..')   # ajuste según la ubicación de su notebook
import pckModelosAvanzados as pm
```

### Dependencias

```
pckEDA             # en la misma carpeta raíz; trae pandas, seaborn, prince, umap-learn
pandas
numpy
matplotlib
scikit-learn
xgboost            # solo para XGBoost (en macOS requiere además: brew install libomp)
scikit-optimize    # opcional: solo para buscar_hiperparametros(metodo='bayes')
```

---

## Inicio rápido

```python
import pckModelosAvanzados as pm

rf = pm.RandomForest(df, target="default", n_estimators=300)
rf.preparar_datos()                    # split 80/20 estratificado
rf.ajustar()                           # entrena
print(rf.evaluar())                    # accuracy, balanced accuracy, P/R/F1 por clase, AUC
print(rf.validacion_cruzada(cv=5))     # media ± desviación (CV estratificada)
rf.plot_matriz_confusion()
```

Comparar varios modelos con el mismo split:

```python
tabla = pm.comparar_modelos([
    pm.KNN(df, "default"),
    pm.RandomForest(df, "default"),
    pm.AdaBoost(df, "default"),
    pm.XGBoost(df, "default"),
], scoring="f1")
print(tabla)
```

---

## Métodos comunes a todos los modelos

| Método | Qué hace |
|---|---|
| `preparar_datos(test_size=0.2, estratificar=True)` | Separa train/test |
| `ajustar()` | Entrena y predice sobre el test |
| `predecir(X)` / `predecir_proba(X)` | Predicciones para datos nuevos |
| `evaluar()` | Accuracy, balanced accuracy, precision/recall/F1 por clase, AUC |
| `indices_binarios()` | VP, VN, FP, FN, sensibilidad, especificidad (2 clases) |
| `validacion_cruzada(cv, scoring)` | CV estratificada: media y desviación |
| `buscar_hiperparametros(param_grid, metodo, cv, scoring, n_iter)` | Grid, Random o Bayes Search |
| `resultados_busqueda()` | Todas las combinaciones probadas |
| `optimizar_umbral(costo_fp, costo_fn)` / `fijar_umbral(u)` | Ajusta el umbral de decisión (2 clases) |
| `plot_matriz_confusion()` / `plot_curva_roc()` | Gráficos de evaluación |
| `importancia_variables()` / `plot_importancia_variables()` | Para modelos basados en árboles |
| `configurar(**hiperparametros)` | Cambia hiperparámetros (luego `ajustar()`) |

Propiedades: `modelo`, `estimador`, `scaler`, `random_state`, `X_train`, `X_test`, `y_train`, `y_test`,
`predicciones`, `probabilidades`, `matriz_confusion`, `reporte`, `mejores_params`,
`hiperparametros`, `umbral`, `clases`, `tipo_aprendizaje`, `tarea`. En `Bagging` y
`RandomForest`, `exactitud_oob` da la exactitud out-of-bag.

### Decisiones de diseño

- **Sin fuga de información:** el escalado vive dentro de un `Pipeline`, así que se
  ajusta solo con el train, incluso dentro de la validación cruzada.
- **Reproducible:** `random_state=42` en el split, la CV y los modelos.
- **Herencia:** `AnalisisDatosExploratorio → Supervisado → Clasificacion → ModeloBase → <modelo>`.
  Hereda `df`, `target`, `X`, `y`, `X_train`... y todos los métodos de limpieza de `pckEDA`
  (`eliminarNulos`, `codificarCategorica`, ...). Las predictoras deben quedar numéricas.
- **Getters y setters:** cada hiperparámetro es una propiedad (`knn.n_neighbors = 7`), igual
  que en `pckEDA`. `configurar(**kwargs)` y `buscar_hiperparametros` los asignan por esa vía.

---

## Estructura del paquete

```
pckModelosAvanzados/
├── README.md                ← este archivo
├── CONTRIBUTING.md          ← guía para añadir modelos (nuevas semanas)
├── CHANGELOG.md             ← historial de cambios del paquete
├── __init__.py
├── base.py                  ← ModeloBase (flujo, métricas, CV, tuneo, umbral, gráficos)
├── comparar.py              ← comparar_modelos
├── knn/            (KNN.md, knn.py, __init__.py)
├── arbol_decision/ (ArbolDecision.md, arbol_decision.py, __init__.py)
├── bagging/        (Bagging.md, bagging.py, __init__.py)
├── random_forest/  (RandomForest.md, random_forest.py, __init__.py)
├── adaboost/       (AdaBoost.md, adaboost.py, __init__.py)
├── gradient_boosting/ (GradientBoosting.md, gradient_boosting.py, __init__.py)
├── xgb/            (XGBoost.md, xgb.py, __init__.py)
├── lda/            (LDA.md, lda.py, __init__.py)
└── qda/            (QDA.md, qda.py, __init__.py)
```

---

## Estándares

- [PEP 8](https://peps.python.org/pep-0008/) — Guía de estilo
- [PEP 20](https://peps.python.org/pep-0020/) — El Zen de Python
- [PEP 257](https://peps.python.org/pep-0257/) — Convenciones de docstrings
- [PEP 328](https://peps.python.org/pep-0328/) — Imports relativos

¿Quiere añadir el modelo de una nueva semana? Consulte [CONTRIBUTING.md](CONTRIBUTING.md).
