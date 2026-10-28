# AdaBoost — Adaptive Boosting (Semana 4)

## ¿Qué hace?

Entrena clasificadores muy simples **uno tras otro**. Después de cada uno, **sube el
peso de los casos que falló** para que el siguiente se concentre en ellos. La decisión
final es un voto donde los clasificadores más acertados pesan más.

**Analogía:** un estudiante que repasa solo las preguntas que falló en el examen anterior.

## ¿Cuándo usarlo?

- Datos **limpios**, cuando quiere potenciar un modelo débil.
- Cuando busca buen desempeño con poca complejidad.

> **Cuidado:** es muy sensible al ruido y a los outliers: les asigna cada vez más peso.
> Limpie los datos antes.

## Uso básico

```python
import pckModelosAvanzados as pm

ada = pm.AdaBoost(df, target="default", n_estimators=100, learning_rate=0.5)
ada.ajustar()
print(ada.evaluar())
ada.plot_curva_iteraciones()
```

## Parámetros

| Parámetro | Por defecto | En palabras simples |
|---|---|---|
| `n_estimators` | `50` | Cantidad de clasificadores simples |
| `learning_rate` | `1.0` | Cuánto cuenta cada uno. Bajo → más lento pero más fino |
| `profundidad_base` | `1` | Tamaño de cada árbol simple (1 = una sola pregunta, "stump") |

## Métodos propios

| Método | Para qué sirve |
|---|---|
| `pesos_clasificadores()` | Error y peso de voto de cada clasificador |
| `curva_iteraciones()` | Exactitud de train y test al ir sumando clasificadores |
| `plot_curva_iteraciones()` | Gráfico de lo anterior: ayuda a detectar sobreajuste |

## Consejos

- Para tunear la profundidad base use la clave `estimator__max_depth`.
- Si `test` baja mientras `train` sube en la curva, reduzca `n_estimators` o `learning_rate`.
