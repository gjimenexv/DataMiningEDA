# QDA — Análisis Discriminante Cuadrático (Semana 6)

**Clasificación:** `Aprendizaje supervisado` · `Clasificación` · Probabilístico / cuadrático · Semana 6

> Usa la variable objetivo (`target`) con las etiquetas reales para aprender; no es un método no supervisado.

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

- Si aparece el error *"The covariance matrix of class ... is not full rank"* hay variables
  muy correlacionadas o redundantes: suba `reg_param` (p. ej. `0.1`) o elimine variables.
- Compare ambos con `pm.comparar_modelos([pm.LDA(df, "y"), pm.QDA(df, "y")])`.
