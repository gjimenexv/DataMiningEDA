# KNN — K-Nearest Neighbors (Semana 1)

**Clasificación:** `Aprendizaje supervisado` · `Clasificación` · Basado en instancias, no paramétrico · Semana 1

> Usa la variable objetivo (`target`) con las etiquetas reales para aprender; no es un método no supervisado.

## ¿Qué hace?

Clasifica una observación según la **clase más común entre sus K vecinos más cercanos**.
No "aprende" una fórmula: guarda los datos y consulta a los vecinos cada vez que predice.

**Analogía:** llegas a una fiesta sin conocer a nadie y miras a las 5 personas más
cercanas. Si 3 hablan de deportes, te unes al grupo de deportes.

## ¿Cuándo usarlo?

- Dataset pequeño o mediano con variables **numéricas**.
- La frontera entre clases puede ser curva (no lineal).
- Como **baseline** rápido antes de probar modelos más complejos.

> **Cuidado:** KNN usa distancias, por eso la clase **escala las variables
> automáticamente** (`scaler='standard'`). No lo desactive salvo que sepa por qué.

## Uso básico

```python
import pckModelosAvanzados as pm

knn = pm.KNN(df, target="default", n_neighbors=5)
knn.preparar_datos()      # separa 80 % train / 20 % test (estratificado)
knn.ajustar()             # entrena y predice sobre el test
print(knn.evaluar())      # accuracy, balanced accuracy, precision/recall/F1 por clase
```

## Parámetros

| Parámetro | Por defecto | En palabras simples |
|---|---|---|
| `n_neighbors` | `5` | Cuántos vecinos votan. K bajo → sobreajusta; K alto → generaliza demasiado |
| `weights` | `'uniform'` | `'distance'` hace que los vecinos más cercanos pesen más |
| `metric` / `p` | `'minkowski'` / `2` | Cómo se mide la cercanía (`p=2` euclidiana, `p=1` Manhattan) |
| `scaler` | `'standard'` | Escalado de variables: `'standard'`, `'minmax'` o `None` |

## Métodos propios

| Método | Para qué sirve |
|---|---|
| `buscar_mejor_k(k_range, cv)` | Prueba varios K con validación cruzada y deja aplicado el mejor |
| `plot_curva_k(k_range, cv)` | Muestra cómo cambia la exactitud con K (el pico es el K ideal) |
| `plot_fronteras_decision(var_x, var_y)` | Dibuja la frontera de decisión usando 2 variables |

Los métodos comunes a todos los modelos (`evaluar`, `validacion_cruzada`,
`buscar_hiperparametros`, `optimizar_umbral`, `plot_matriz_confusion`,
`plot_curva_roc`...) están en el [README](../README.md#métodos-comunes-a-todos-los-modelos).

## Ejemplo completo

```python
knn = pm.KNN(df, target="default")
mejor_k = knn.buscar_mejor_k(range(1, 31), cv=5)   # elige K
knn.ajustar()
knn.plot_curva_k()
knn.plot_matriz_confusion()
print(knn.validacion_cruzada())                     # media ± desviación
```

## Consejos

- Si el accuracy es alto pero el recall de la clase minoritaria es bajo, mire la
  matriz de confusión y pruebe `optimizar_umbral()`.
- KNN es lento al predecir con muchos datos y sufre con muchas variables
  (*maldición de la dimensionalidad*): elimine variables poco útiles.
