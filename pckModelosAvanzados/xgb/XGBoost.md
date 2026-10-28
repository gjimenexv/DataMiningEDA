# XGBoost — Extreme Gradient Boosting (Semana 5)

## ¿Qué hace?

Es Gradient Boosting "mejor jugado": **más prudente** (desconfía de hojas con pocos
datos), **poda cortes que no valen la pena**, **maneja valores faltantes solo** y es
muy rápido.

## Instalación

```
pip install xgboost
```

Si no está instalado, la clase avisa con un mensaje claro al ajustar.

> **Requisito:** las etiquetas del target deben ser `0, 1, ...` (no texto).

## Uso básico

```python
import pckModelosAvanzados as pm

xg = pm.XGBoost(df, target="churn", n_estimators=500, learning_rate=0.05,
                early_stopping_rounds=20, scale_pos_weight="auto")
xg.ajustar()
print(xg.evaluar())
print("Árboles usados:", xg.arboles_usados)
xg.plot_curva_aprendizaje()
```

## Parámetros

| Parámetro | Por defecto | En palabras simples |
|---|---|---|
| `n_estimators` | `300` | Máximo de árboles (use muchos + early stopping) |
| `learning_rate` | `0.1` | Tamaño de cada corrección (0.01–0.1) |
| `max_depth` | `4` | Preguntas por árbol (3–6) |
| `reg_lambda` | `1.0` | Desconfianza hacia hojas con pocos datos (1–10) |
| `gamma` | `0.0` | Costo de cada corte extra: más alto → árboles más simples |
| `subsample` | `1.0` | Fracción de filas por árbol (0.7–1) |
| `scale_pos_weight` | `None` | Peso de la clase rara. `'auto'` = negativos / positivos |
| `early_stopping_rounds` | `None` | Detiene si la validación no mejora en ese número de árboles |

## Métodos y propiedades propios

| Nombre | Para qué sirve |
|---|---|
| `arboles_usados` | Cuántos árboles se usaron realmente tras el early stopping |
| `plot_curva_aprendizaje()` | Log-loss de train y validación por árbol |

## Consejos

- Con clases desbalanceadas no confíe en el accuracy: use `scoring="f1"` o AUC,
  `scale_pos_weight` y `optimizar_umbral()`.
- Con early stopping, el modelo reserva el 20 % del train como validación interna.
