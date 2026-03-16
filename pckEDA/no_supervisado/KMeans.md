# KMeans — Clustering K-Means

## ¿Qué hace?

**K-Means** divide los datos en `k` grupos (clusters) minimizando la distancia de cada
observación al centroide (centro) de su grupo. El algoritmo alterna entre dos pasos:
1. Asigna cada punto al centroide más cercano.
2. Recalcula los centroides como la media de los puntos asignados.

Repite hasta que los clusters se estabilizan.

**Analogía:** Imagina que quieres repartir 10,000 clientes en 4 grupos. K-Means elige
4 "representantes" al azar, asigna cada cliente al más parecido, actualiza los representantes,
y repite hasta que los grupos dejan de cambiar.

---

## ¿Cuándo usarlo?

- Cuando tienes un dataset **grande** y necesitas velocidad.
- Cuando sospechas que los clusters son **esféricos y de tamaño similar**.
- Cuando quieres un resultado **reproducible** con `random_state` fijo.
- Como complemento a t-SNE o UMAP para colorear proyecciones.

> **Requisito:** Todos los datos deben ser **numéricos**. Los datos se estandarizan automáticamente.

---

## Flujo de uso

```
mf.KMeans(path, num)        # 1. Cargar datos desde CSV
    ↓ métodos EDA heredados  # 2. Limpiar y preparar
km.ajustar()                # 3. Entrenar el modelo
km.plot_*()                 # 4. Visualizar resultados
```

## Creación del modelo

```python
import pckEDA as mf

km = mf.KMeans('hotel_bookings.csv', 1, n_clusters=4)
```

### Parámetros del constructor

| Parámetro | Tipo | Valor por defecto | Descripción sencilla |
|-----------|------|-------------------|----------------------|
| `path` | `str` | — | Ruta al archivo CSV |
| `num` | `int` | — | Formato del CSV: **1** = coma con índice, **2** = punto y coma sin índice |
| `n_clusters` | `int` | `3` | ¿En cuántos grupos dividir los datos? |
| `init` | `str` | `'k-means++'` | Método de inicialización de centroides (ver tabla abajo) |
| `n_init` | `int` | `10` | Número de ejecuciones con distintas semillas — se conserva la mejor |
| `random_state` | `int` | `42` | Semilla para reproducibilidad |

#### Método de inicialización (`init`)

| Valor | Descripción | ¿Cuándo usarlo? |
|-------|-------------|-----------------|
| `'k-means++'` | Inicializa centroides espaciados — más estable | **Recomendado siempre** |
| `'random'` | Inicialización aleatoria | Solo para experimentar |

### Método `ajustar()`

Entrena el modelo sobre `self.df`. Debe llamarse después de codificar categóricas,
eliminar nulos y conservar solo columnas numéricas.

---

## Propiedades

| Propiedad | Qué contiene |
|-----------|--------------|
| `km.etiquetas` | Array con el número de cluster de cada observación (0-indexado) |
| `km.centroides` | Matriz de centroides en el espacio escalado (n_clusters × n_features) |
| `km.inercia` | Suma de distancias cuadradas al centroide — menor es mejor |
| `km.silhouette` | Coeficiente de Silhouette promedio (−1 a 1) |
| `km.resumen` | DataFrame con la media de cada variable por cluster |
| `km.modelo` | Modelo `sklearn.KMeans` ajustado |

### ¿Cómo interpretar el Silhouette?

| Valor | Interpretación |
|-------|----------------|
| ≥ 0.70 | Excelente separación entre clusters |
| 0.50 – 0.69 | Separación razonable |
| 0.25 – 0.49 | Clusters solapados — considera cambiar `k` |
| < 0.25 | Clusters no diferenciados — revisa los datos |

---

## Métodos de visualización

### `km.plot_codo(max_clusters=10, titulo='...')`
Gráfico del método del codo: inercia vs número de clusters.
El punto de inflexión ("codo") sugiere el `k` óptimo.

### `km.plot_silhouette(max_clusters=10, titulo='...')`
Coeficiente de Silhouette para cada valor de `k` entre 2 y `max_clusters`.
El pico más alto indica el `k` óptimo.

### `km.plot_mapa_calor(titulo='...')`
Heatmap normalizado de la media de cada variable por cluster.

### `km.plot_distribucion(titulo='...')`
Barras con el número de observaciones por cluster.

### `km.plot_dispersion(col_x, col_y, titulo=None)`
Diagrama de dispersión de dos variables coloreado por cluster.

---

## Ejemplo completo

```python
import matplotlib.pyplot as plt
import pckEDA as mf

# 1. Cargar y preparar
km = mf.KMeans('hotel_bookings.csv', 1, n_clusters=4)
km.codificarCategorica('hotel')
km.codificarCategorica('deposit_type', mapeo={'No Deposit': 0, 'Non Refund': 1, 'Refundable': 2})
km.eliminarNulos()
km.analisisNumerico()

# 2. Ajustar
km.ajustar()
print(f"Silhouette: {km.silhouette:.4f}")
print(f"Inercia   : {km.inercia:.2f}")

# 3. Determinar k óptimo
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plt.sca(axes[0]); km.plot_codo()
plt.sca(axes[1]); km.plot_silhouette()
plt.tight_layout(); plt.show()

# 4. Visualizaciones
plt.figure(figsize=(14, 5)); km.plot_mapa_calor()
plt.figure(figsize=(8, 5));  km.plot_distribucion()
plt.figure(figsize=(10, 6)); km.plot_dispersion('adr', 'lead_time')
plt.show()
```

---

## Consejos para no estadísticos

- **¿Cuántos clusters usar?** Usa `plot_codo()` y `plot_silhouette()` antes de decidir — el número donde ambos "se doblan" o alcanzan su pico es el óptimo.
- **K-Means no es determinista** si no fijas `random_state` — el mismo código puede dar resultados distintos en distintas ejecuciones. El valor por defecto (`42`) garantiza reproducibilidad.
- **El mapa de calor** es la herramienta más útil para *nombrar* los clusters en términos de negocio (e.g. "Cluster 0 = reservas económicas de última hora").
- **Limitación clave:** K-Means supone clusters esféricos. Si tus datos tienen formas irregulares, considera HAC.
