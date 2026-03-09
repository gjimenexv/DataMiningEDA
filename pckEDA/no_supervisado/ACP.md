# ACP — Análisis de Componentes Principales (PCA)

## ¿Qué hace?

El **ACP** (o PCA en inglés) es una técnica de **reducción de dimensionalidad**. Su objetivo es simplificar un dataset con muchas variables en un número menor de "resúmenes" llamados **componentes principales**, conservando la mayor cantidad de información posible.

**Analogía:** Imagina que tienes 20 preguntas de una encuesta. El ACP descubre que, en realidad, la mayoría de las respuestas se pueden resumir en 3 o 4 "temas" principales — y trabaja con esos temas en lugar de las 20 preguntas originales.

---

## ¿Cuándo usarlo?

- Cuando tu dataset tiene **muchas variables** y quieres visualizarlo en 2D o 3D.
- Cuando sospechas que varias variables están muy correlacionadas (miden cosas parecidas).
- Como paso previo a otros algoritmos para reducir ruido y mejorar rendimiento.
- Para identificar qué variables son las más importantes en el dataset.

> **Requisito:** Todos los datos deben ser **numéricos**. Codifica las variables de texto antes de usar ACP.

---

## Creación del modelo

```python
import pckEDA as mf

acp = mf.ACP(datos, n_componentes=5)
```

### Parámetros del constructor

| Parámetro | Tipo | Valor por defecto | Descripción sencilla |
|-----------|------|-------------------|----------------------|
| `datos` | `DataFrame` | — | Tabla de datos numéricos (sin columnas de texto) |
| `n_componentes` | `int` | `5` | ¿Cuántos componentes principales quieres calcular? |

> **¿Cuántos componentes usar?** Empieza con 5. Revisa `acp.var_explicada` — si los primeros 2 o 3 componentes ya explican más del 70% de la varianza, son suficientes.

---

## Propiedades (resultados del modelo)

| Propiedad | Qué contiene |
|-----------|--------------|
| `acp.var_explicada` | Lista con el % de varianza explicado por cada componente |
| `acp.correlacion_var` | Correlación de cada variable original con cada componente |
| `acp.coordenadas_ind` | Posición de cada observación en el espacio de componentes |
| `acp.contribucion_ind` | Cuánto "peso" tiene cada observación en cada componente |
| `acp.cos2_ind` | Calidad de representación de cada observación (0 a 1) |
| `acp.modelo` | El modelo Prince ajustado (para usuarios avanzados) |
| `acp.datos` | El DataFrame original |

---

## Métodos de visualización

### `acp.plot_plano_principal(ejes=[0,1], ind_labels=True, titulo='...')`

Muestra la posición de cada observación en el plano formado por dos componentes.

| Parámetro | Descripción | Valores típicos |
|-----------|-------------|-----------------|
| `ejes` | Qué dos componentes graficar | `[0,1]` (1º y 2º), `[2,3]` (3º y 4º) |
| `ind_labels` | ¿Mostrar el ID de cada punto? | `True` / `False` (usa `False` con datasets grandes) |
| `titulo` | Título del gráfico | Cualquier texto |

---

### `acp.plot_circulo(ejes=[0,1], var_labels=True, titulo='...')`

El **círculo de correlación** muestra cómo se relacionan las variables originales con los componentes.

- Flechas largas → la variable está bien representada en ese plano.
- Flechas en la misma dirección → variables correlacionadas positivamente.
- Flechas opuestas → variables correlacionadas negativamente.

| Parámetro | Descripción | Valores típicos |
|-----------|-------------|-----------------|
| `ejes` | Qué dos componentes graficar | `[0,1]`, `[2,3]` |
| `var_labels` | ¿Mostrar nombre de las variables? | `True` / `False` |
| `titulo` | Título del gráfico | Cualquier texto |

---

### `acp.plot_sobreposicion(ejes=[0,1], ind_labels=True, var_labels=True, titulo='...')`

Combina el plano principal y el círculo de correlación en un solo gráfico (biplot).

| Parámetro | Descripción |
|-----------|-------------|
| `ejes` | Qué dos componentes graficar |
| `ind_labels` | ¿Mostrar IDs de observaciones? |
| `var_labels` | ¿Mostrar nombres de variables? |
| `titulo` | Título del gráfico |

---

## Ejemplo completo

```python
import pandas as pd
import matplotlib.pyplot as plt
import pckEDA as mf

# Cargar y limpiar datos
datos = pd.read_csv("hotel_bookings.csv", index_col=0)
datos = datos.select_dtypes(include="number").dropna()

# Crear modelo
acp = mf.ACP(datos, n_componentes=5)

# ¿Cuánta información conserva cada componente?
print(acp.var_explicada)

# Gráficos en un par de ejes (componentes 1 y 2)
plt.figure(figsize=(10, 7))
acp.plot_plano_principal(ejes=[0, 1], ind_labels=False)

plt.figure(figsize=(8, 8))
acp.plot_circulo(ejes=[0, 1])

plt.figure(figsize=(10, 8))
acp.plot_sobreposicion(ejes=[0, 1], ind_labels=False)
plt.show()
```

---

## Consejos para no estadísticos

- **Varianza explicada:** Si el componente 1 explica el 40% y el componente 2 el 25%, juntos capturan el 65% de toda la información del dataset. Cuanto más alto, mejor.
- **Círculo de correlación:** Las variables cuyas flechas apuntan en la misma dirección "van de la mano". Variables en ángulo recto son independientes.
- **Plano principal:** Las observaciones cercanas entre sí son parecidas. Las agrupaciones visibles sugieren la existencia de clusters naturales.
- **cos² bajo:** Si un punto está cerca del origen en el plano principal, significa que ese individuo no se representa bien en esos dos componentes — prueba con otros ejes.
