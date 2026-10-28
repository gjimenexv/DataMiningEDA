# RandomForest — Bosque Aleatorio (Semana 3)

**Clasificación:** `Aprendizaje supervisado` · `Clasificación` · Ensamble paralelo (bagging) · Semana 3

> Usa la variable objetivo (`target`) con las etiquetas reales para aprender; no es un método no supervisado.

## ¿Qué hace?

Igual que Bagging (muchos árboles + votación), pero cada árbol, en cada pregunta,
**solo puede mirar un subconjunto al azar de las variables**. Eso hace que los árboles
sean distintos entre sí y que su votación sea más confiable.

**Analogía:** un jurado donde cada miembro ve solo parte del expediente: opinan
distinto, pero el veredicto conjunto es más robusto.

## ¿Cuándo usarlo?

- Es una excelente primera opción con **poco tuneo**: robusto y estable.
- No requiere escalar variables y tolera ruido y outliers.
- Para saber qué variables importan (`plot_importancia_variables`).

> **Costo:** es más lento y menos interpretable que un solo árbol.

## Uso básico

```python
import pckModelosAvanzados as pm

rf = pm.RandomForest(df, target="default", n_estimators=300, class_weight="balanced")
rf.ajustar()
print(rf.evaluar())
print("Exactitud out-of-bag:", rf.exactitud_oob)
rf.plot_importancia_variables(top=10)
```

## Parámetros

| Parámetro | Por defecto | En palabras simples |
|---|---|---|
| `n_estimators` | `100` | Número de árboles (100–500). Más no daña, pero rinde cada vez menos |
| `max_depth` | `None` | Profundidad máxima de cada árbol |
| `max_features` | `'sqrt'` | Variables que cada árbol puede mirar en cada pregunta |
| `min_samples_leaf` | `1` | Mínimo de casos por hoja |
| `class_weight` | `None` | `'balanced'` si hay una clase rara |
| `oob_score` | `True` | Calcula una exactitud "gratis" con las filas no usadas por cada árbol |

## Propiedades propias

- `rf.exactitud_oob`: exactitud out-of-bag.
- `rf.importancia_variables()`: Series ordenada con la importancia de cada variable.

## Ejemplo con tuneo

```python
mejores = rf.buscar_hiperparametros(
    {"n_estimators": [100, 300], "max_depth": [5, 10, None], "max_features": ["sqrt", 0.5]},
    metodo="random", n_iter=10, scoring="f1")
print(mejores)
```

## Consejos

- Para el Credit Default Challenge priorice `scoring="f1"` y revise el recall de la clase 1.
