# GradientBoosting — Boosting por gradiente (Semana 5)

**Clasificación:** `Aprendizaje supervisado` · `Clasificación` · Ensamble secuencial (boosting por gradiente) · Semana 5

> Usa la variable objetivo (`target`) con las etiquetas reales para aprender; no es un método no supervisado.

## ¿Qué hace?

Empieza con una predicción simple y le va **sumando pequeñas correcciones**. Cada
árbol nuevo no predice la clase, sino **lo que todavía falta** (el error del modelo
hasta ese momento). Se suma multiplicado por una tasa de aprendizaje pequeña.

**Analogía:** un golfista que en cada golpe apunta a lo que le falta para llegar al hoyo.

## ¿Cuándo usarlo?

- Datos tabulares donde busca **máxima precisión**.
- Cuando puede dedicar tiempo a afinar hiperparámetros.

> **Cuidado:** con demasiados árboles memoriza los casos raros. Use una tasa de
> aprendizaje baja y `n_iter_no_change` (*early stopping*).

## Uso básico

```python
import pckModelosAvanzados as pm

gb = pm.GradientBoosting(df, target="default", n_estimators=300,
                         learning_rate=0.05, n_iter_no_change=10)
gb.ajustar()
print(gb.evaluar())
gb.plot_curva_iteraciones()
```

## Parámetros

| Parámetro | Por defecto | En palabras simples |
|---|---|---|
| `n_estimators` | `100` | Máximo de árboles |
| `learning_rate` | `0.1` | Tamaño de cada corrección (0.01–0.1) |
| `max_depth` | `3` | Preguntas por árbol (3–6) |
| `subsample` | `1.0` | Fracción de filas por árbol; menos de 1 reduce el sobreajuste |
| `n_iter_no_change` | `None` | Detiene el entrenamiento si no mejora en ese número de árboles |

## Métodos propios

| Método | Para qué sirve |
|---|---|
| `curva_iteraciones()` | Exactitud de train y test al ir sumando árboles |
| `plot_curva_iteraciones()` | Muestra dónde conviene detenerse |

## Diagnóstico rápido

- **Sobreajusta:** menos profundidad, tasa más pequeña, `subsample < 1`.
- **Subajusta:** más árboles o más profundidad.
