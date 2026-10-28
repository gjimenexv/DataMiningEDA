# Bagging — Bootstrap Aggregating (Semana 3)

**Clasificación:** `Aprendizaje supervisado` · `Clasificación` · Ensamble paralelo (bagging) · Semana 3

> Usa la variable objetivo (`target`) con las etiquetas reales para aprender; no es un método no supervisado.

## ¿Qué hace?

Entrena **muchos árboles**, cada uno con una muestra distinta (con reemplazo) de los
datos, y decide por **votación**. Un solo árbol cambia mucho si cambian los datos;
el promedio de muchos es mucho más estable.

**Analogía:** en lugar de preguntarle a un solo experto, consulta a 100 y se queda
con la opinión mayoritaria.

## ¿Cuándo usarlo?

- Cuando un árbol individual sobreajusta (alta varianza).
- Como paso intermedio para entender Random Forest, que añade un sorteo de variables.

## Uso básico

```python
import pckModelosAvanzados as pm

bag = pm.Bagging(df, target="default", n_estimators=100)
bag.ajustar()
print(bag.evaluar())
print("Exactitud out-of-bag:", bag.exactitud_oob)
```

## Parámetros

| Parámetro | Por defecto | En palabras simples |
|---|---|---|
| `n_estimators` | `100` | Cantidad de árboles |
| `max_depth` | `None` | Profundidad de cada árbol |
| `max_samples` | `1.0` | Fracción de filas de cada muestra |
| `oob_score` | `True` | Estima el error con las filas que cada árbol no vio (~37 %) |

## Propiedades propias

- `bag.exactitud_oob`: estimación del accuracy sin necesitar un set de validación.

## Consejos

- Para tunear la profundidad use la clave `estimator__max_depth`:
  `bag.buscar_hiperparametros({"estimator__max_depth": [3, 5, None]})`.
- En la tarea de la Semana 3, Bagging de 100 árboles fue lo que más estabilizó el árbol.
