# LDA — Análisis Discriminante Lineal (Semana 6)

**Clasificación:** `Aprendizaje supervisado` · `Clasificación` · Probabilístico / lineal (también reduce dimensionalidad) · Semana 6

> Usa la variable objetivo (`target`) con las etiquetas reales para aprender; no es un método no supervisado.

## ¿Qué hace?

Busca la **línea (o plano) sobre la cual proyectar los datos para que las clases
queden lo más separadas posible**, con cada clase lo más compacta posible. Clasifica
cada punto según la clase cuyo centro proyectado le queda más cerca. También sirve
para **reducir dimensiones** (hasta K − 1 ejes, con K clases).

## ¿Cuándo usarlo?

- Clases aproximadamente "en forma de campana" con dispersión similar.
- Pocos datos o necesidad de un modelo simple e interpretable.
- Para visualizar clases en 2D (`plot_proyeccion`).

> **No es buena idea si:** la frontera es curva, hay muchos outliers o las clases
> tienen dispersiones muy distintas (pruebe [QDA](../qda/QDA.md)).

## Uso básico

```python
import pckModelosAvanzados as pm

lda = pm.LDA(df, target="especie")
lda.ajustar()
print(lda.evaluar())
lda.plot_proyeccion()
print(lda.varianza_explicada)
```

## Parámetros

| Parámetro | Por defecto | En palabras simples |
|---|---|---|
| `solver` | `'svd'` | Método de cálculo. `'lsqr'` o `'eigen'` permiten `shrinkage` |
| `shrinkage` | `None` | Regularización: `'auto'` ayuda con muchas variables o pocos datos |
| `scaler` | `'standard'` | LDA no es invariante a la escala: se recomienda estandarizar |

## Métodos propios

| Método | Para qué sirve |
|---|---|
| `transformar(X)` | Proyecta datos sobre los ejes discriminantes `LD1`, `LD2`... |
| `varianza_explicada` | Cuánta separación aporta cada eje |
| `covarianzas_por_clase()` | Revisa el supuesto de igual dispersión entre clases |
| `plot_proyeccion()` | Grafica el test sobre los ejes discriminantes |

## Consejos

- Si `covarianzas_por_clase()` muestra determinantes muy diferentes entre clases,
  el supuesto de LDA no se cumple: compare con QDA.
