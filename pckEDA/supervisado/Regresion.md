# Regresion — Regresión Supervisada (en desarrollo)

## Estado actual

> ⚠️ Esta clase es actualmente un **esqueleto** (placeholder). Los métodos del algoritmo aún no están implementados.

---

## ¿Qué hará cuando esté completa?

La clase `Regresion` implementará algoritmos de **aprendizaje supervisado** para predecir un **valor numérico continuo** a partir de otras variables.

**Analogía:** Conoces las características de una reserva de hotel (tipo de hotel, época del año, número de noches, etc.) y quieres predecir la **tarifa diaria (`adr`)** que pagará el cliente. La regresión aprende esa relación de miles de ejemplos históricos.

### Clasificación vs Regresión

| | Clasificación | Regresión |
|-|---------------|-----------|
| ¿Qué predice? | Una categoría ("Cancelado" / "No cancelado") | Un número (tarifa, precio, tiempo) |
| Ejemplo | ¿Se cancelará esta reserva? | ¿Cuánto pagará este cliente? |
| Métrica principal | Exactitud, F1 | Error cuadrático medio (RMSE), R² |

---

## Creación del objeto (estructura actual)

```python
import pckEDA as mf

regresion = mf.Regresion(df, target="adr")
```

### Parámetros del constructor

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `df` | `DataFrame` | Tabla de datos con la columna objetivo incluida |
| `target` | `str` | Nombre de la columna numérica a predecir |

> **Columna objetivo (`target`):** Debe ser una variable numérica continua (ej. precio, temperatura, duración).

---

## Algoritmos previstos

| Algoritmo | Cuándo usar |
|-----------|-------------|
| Regresión lineal | Relaciones lineales; muy interpretable |
| Ridge / Lasso | Cuando hay muchas variables; controla el sobreajuste |
| Random Forest Regressor | Alta precisión; captura relaciones no lineales |
| Gradient Boosting | Muy alta precisión; requiere más ajuste de parámetros |

---

## Métricas de evaluación previstas

| Métrica | ¿Qué mide? | Interpretación |
|---------|------------|----------------|
| MAE | Error absoluto medio | En las mismas unidades que `target`; fácil de interpretar |
| RMSE | Raíz del error cuadrático medio | Penaliza más los errores grandes |
| R² | Coeficiente de determinación | Entre 0 y 1; cuanto más cercano a 1, mejor el ajuste |

---

## Guía para contribuir

Para implementar esta clase, sigue el patrón de `acp.py` y `hac.py`:

1. Hereda de `Supervisado`
2. Separa features (`X`) y variable objetivo (`y`) a partir de `target`
3. Usa `sklearn.model_selection.train_test_split` para dividir train/test
4. Expón resultados como `@property` con name-mangling
5. Implementa al menos: entrenamiento, predicción, métricas (MAE, RMSE, R²) y gráfico de residuos
6. Registra en los `__init__.py` (ver `pckEDA/README.md`)
