# ArbolDecision — Árbol de decisión CART (Semana 2)

**Clasificación:** `Aprendizaje supervisado` · `Clasificación` · Basado en reglas · Semana 2

> Usa la variable objetivo (`target`) con las etiquetas reales para aprender; no es un método no supervisado.

## ¿Qué hace?

Hace una serie de preguntas de sí/no sobre las variables (*¿el monto es mayor a 660?*)
hasta llegar a una decisión. Cada pregunta se elige para que los grupos resultantes
queden lo más "puros" posible (medido con el **índice de Gini**).

**Analogía:** el juego de las 20 preguntas, donde siempre se hace primero la pregunta
que más descarta.

## ¿Cuándo usarlo?

- Cuando necesita **explicar las reglas** al negocio.
- Cuando hay mezcla de escalas: **no requiere escalar** las variables.
- Como pieza base de Random Forest y los modelos de boosting.

> **Cuidado:** un árbol sin límites memoriza los datos (sobreajuste). Limite
> `max_depth` o use `ccp_alpha` para podarlo.

## Uso básico

```python
import pckModelosAvanzados as pm

arbol = pm.ArbolDecision(df, target="default", max_depth=4)
arbol.ajustar()
print(arbol.evaluar())
print(arbol.reglas_texto())     # las reglas en lenguaje de "si... entonces"
arbol.plot_arbol(max_depth=3)
```

## Parámetros

| Parámetro | Por defecto | En palabras simples |
|---|---|---|
| `criterion` | `'gini'` | Cómo se mide la pureza: `'gini'`, `'entropy'`, `'log_loss'` |
| `max_depth` | `None` | Máximo de preguntas seguidas. `None` = sin límite (riesgo de sobreajuste) |
| `min_samples_split` | `2` | Mínimo de casos para permitir dividir un grupo |
| `min_samples_leaf` | `1` | Mínimo de casos en cada hoja final |
| `ccp_alpha` | `0.0` | Poda: mientras mayor, más simple el árbol |
| `class_weight` | `None` | `'balanced'` si una clase es mucho más rara que la otra |

## Métodos propios

| Método | Para qué sirve |
|---|---|
| `gini(y)` | Calcula la impureza de un grupo (0 = puro, 0.5 = máximo desorden en 2 clases) |
| `gini_ponderado(y_izq, y_der)` | Calidad de una división: lo que CART intenta minimizar |
| `mejor_umbral(variable)` | Encuentra el mejor corte de una variable numérica (ejemplo de clase) |
| `reglas_texto()` | Imprime el árbol como reglas legibles |
| `ruta_poda()` | Lista de valores de `ccp_alpha` posibles para podar |
| `plot_arbol(max_depth)` | Dibuja el árbol |

## Ejemplo: entender un corte a mano

```python
pm.ArbolDecision.gini([0] * 16 + [1] * 8)        # 0.444: nodo raíz del ejemplo
arbol.mejor_umbral("monto")                       # (660.0, gini_ponderado)
```

## Consejos

- Compare el desempeño de train y test: una gran diferencia indica sobreajuste.
- Para podar con criterio: `buscar_hiperparametros({"ccp_alpha": [0, .001, .01]})`.
