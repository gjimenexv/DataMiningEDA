# QDA — Análisis Discriminante Cuadrático (Semana 6)

## ¿Qué hace?

Es como [LDA](../lda/LDA.md), pero **permite que cada clase tenga su propia forma y
dispersión**. Por eso la frontera entre clases puede ser **curva** (cuadrática).

## ¿Cuándo usarlo?

- Cuando las clases tienen dispersiones claramente distintas.
- Cuando LDA queda corto porque la frontera real no es una línea recta.

> **Cuidado:** estima muchos más parámetros que LDA; con pocos datos sobreajusta.
> Suba `reg_param` para estabilizarlo.

## Uso básico

```python
import pckModelosAvanzados as pm

qda = pm.QDA(df, target="especie", reg_param=0.1)
qda.ajustar()
print(qda.evaluar())
qda.plot_matriz_confusion()
```

## Parámetros

| Parámetro | Por defecto | En palabras simples |
|---|---|---|
| `reg_param` | `0.0` | Regularización de las covarianzas (0–1). Reduce el sobreajuste |
| `scaler` | `'standard'` | Escalado de variables (recomendado) |

## LDA vs. QDA

| | LDA | QDA |
|---|---|---|
| Dispersión de las clases | Igual para todas | Propia de cada clase |
| Frontera | Recta | Curva |
| Riesgo de sobreajuste | Menor | Mayor |
| Datos necesarios | Pocos | Más |

## Consejos

- Compare ambos con `pm.comparar_modelos([pm.LDA(df, "y"), pm.QDA(df, "y")])`.
