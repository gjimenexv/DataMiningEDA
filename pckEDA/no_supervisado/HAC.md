# HAC — Clustering Jerárquico Aglomerativo

## ¿Qué hace?

El **HAC** agrupa observaciones similares sin que tú le indiques de antemano a qué grupo pertenece cada una. Lo hace de forma jerárquica: empieza tratando cada observación como un grupo propio y va fusionando los más parecidos hasta que todo forma un único árbol de relaciones llamado **dendrograma**.

**Analogía:** Imagina que tienes 100 clientes de un hotel. El HAC los va "emparejando" de menor a mayor similitud, como un torneo: primero los más parecidos forman duplas, luego las duplas forman cuartetos, y así hasta que todos están en un único grupo. Tú decides dónde "cortar" ese árbol para obtener el número de segmentos que quieres.

---

## ¿Cuándo usarlo?

- Cuando quieres **segmentar** observaciones sin saber cuántos grupos existen de antemano.
- Cuando el dendrograma ayuda a **justificar visualmente** el número de clusters.
- Cuando necesitas un resultado reproducible (a diferencia de K-Means, HAC es determinista).
- Con datasets de **tamaño mediano** (hasta ~10,000 filas). Con datasets muy grandes puede ser lento.

> **Requisito:** Todos los datos deben ser **numéricos**. Los datos se estandarizan automáticamente.

---

## Creación del modelo

```python
import pckEDA as mf

hac = mf.HAC(datos, n_clusters=4, metodo='ward', metrica='euclidean')
```

### Parámetros del constructor

| Parámetro | Tipo | Valor por defecto | Descripción sencilla |
|-----------|------|-------------------|----------------------|
| `datos` | `DataFrame` | — | Tabla de datos numéricos ya limpios |
| `n_clusters` | `int` | `3` | ¿En cuántos grupos quieres dividir los datos? |
| `metodo` | `str` | `'ward'` | Criterio para decidir qué grupos fusionar (ver tabla abajo) |
| `metrica` | `str` | `'euclidean'` | Cómo medir la "distancia" entre observaciones |

#### Métodos de enlace (`metodo`)

| Valor | Criterio de fusión | ¿Cuándo usarlo? |
|-------|--------------------|-----------------|
| `'ward'` | Minimiza la varianza dentro de cada cluster | **Recomendado para la mayoría de casos** — produce clusters compactos y equilibrados |
| `'complete'` | Usa la distancia máxima entre pares | Clusters compactos; sensible a outliers |
| `'average'` | Usa la distancia promedio entre pares | Equilibrio entre `ward` y `single` |
| `'single'` | Usa la distancia mínima entre pares | Detecta clusters alargados; muy sensible a ruido |

> **Regla general:** Usa `'ward'` salvo que tengas razones específicas para cambiar.

#### Métrica de distancia (`metrica`)

| Valor | ¿Qué mide? |
|-------|------------|
| `'euclidean'` | Distancia "en línea recta" entre dos puntos — **la más común** |
| `'cosine'` | Ángulo entre vectores — útil para datos de texto o frecuencias |
| `'manhattan'` | Suma de diferencias absolutas — más robusta a outliers |

> **Nota:** Cuando `metodo='ward'`, scipy ignora el parámetro `metrica` y siempre usa distancia euclídea.

---

## Propiedades (resultados del modelo)

| Propiedad | Qué contiene |
|-----------|--------------|
| `hac.etiquetas` | Array con el número de cluster de cada observación (1, 2, 3, ...) |
| `hac.resumen` | Tabla con la media de cada variable por cluster |
| `hac.cophenet_corr` | Número entre 0 y 1 que indica la calidad del clustering |
| `hac.enlace` | Matriz de enlace interna de scipy (usuarios avanzados) |
| `hac.datos_escalados` | Datos normalizados que se usaron internamente |
| `hac.n_clusters` | Número de clusters configurado |
| `hac.metodo` | Método de enlace utilizado |
| `hac.metrica` | Métrica de distancia utilizada |

### ¿Cómo interpretar `cophenet_corr`?

| Valor | Interpretación |
|-------|----------------|
| ≥ 0.80 | Excelente — la jerarquía representa muy bien los datos |
| 0.70 – 0.79 | Bueno — aceptable para la mayoría de análisis |
| < 0.70 | Regular — considera cambiar el método de enlace |

---

## Métodos de visualización

### `hac.plot_dendrograma(max_hojas=30, color_umbral=None, titulo='...')`

Muestra el árbol jerárquico de fusiones. La altura de cada unión indica cuán distintos eran los grupos al fusionarse.

| Parámetro | Descripción | Valores típicos |
|-----------|-------------|-----------------|
| `max_hojas` | Máximo de nodos visibles en la base del árbol | `20`–`50`. Con datasets grandes, usa valores pequeños para mayor claridad |
| `color_umbral` | Altura de corte para colorear ramas | `None` (automático) o un número concreto |
| `titulo` | Título del gráfico | Cualquier texto |

> **¿Cómo leer el dendrograma?** Busca el "salto" más grande en altura — ahí es donde el árbol sugiere naturalmente cuántos clusters existen. Si hay un salto muy grande antes de la última fusión, probablemente hay 2 grupos bien diferenciados.

---

### `hac.plot_mapa_calor(titulo='...')`

Muestra la media de cada variable por cluster, normalizada. Permite identificar el **perfil** de cada cluster.

- **Verde intenso** → ese cluster tiene valores altos en esa variable.
- **Rojo intenso** → ese cluster tiene valores bajos.

> **Tip:** Este gráfico es la herramienta más útil para *nombrar* los clusters (e.g. "Cluster 1 = reservas de lujo", "Cluster 3 = reservas de último minuto").

---

### `hac.plot_distribucion(titulo='...')`

Barras con la cantidad de observaciones en cada cluster. Ayuda a detectar clusters muy pequeños (posiblemente outliers) o muy desbalanceados.

---

### `hac.plot_dispersion(col_x, col_y, titulo=None)`

Diagrama de dispersión entre dos variables, coloreando cada punto según su cluster. Muestra visualmente qué tan bien separados están los grupos.

| Parámetro | Descripción |
|-----------|-------------|
| `col_x` | Nombre de la columna para el eje horizontal |
| `col_y` | Nombre de la columna para el eje vertical |
| `titulo` | Título (si es `None`, se genera automáticamente) |

---

## Ejemplo completo

```python
import pandas as pd
import matplotlib.pyplot as plt
import pckEDA as mf

# Cargar y preparar datos
datos = pd.read_csv("hotel_bookings.csv", index_col=0)
datos = datos.select_dtypes(include="number").dropna()

# Crear modelo con 4 clusters usando Ward
hac = mf.HAC(datos, n_clusters=4, metodo='ward')

# Calidad del clustering
print(f"Correlación cofenética: {hac.cophenet_corr:.4f}")

# Visualizaciones
plt.figure(figsize=(14, 6))
hac.plot_dendrograma(max_hojas=40)

plt.figure(figsize=(14, 5))
hac.plot_mapa_calor()

plt.figure(figsize=(7, 5))
hac.plot_distribucion()

plt.figure(figsize=(10, 6))
hac.plot_dispersion('adr', 'lead_time')

plt.show()

# Perfil estadístico de cada cluster
print(hac.resumen)
```

---

## Comparar métodos de enlace

```python
metodos = ['ward', 'complete', 'average', 'single']
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for ax, metodo in zip(axes.ravel(), metodos):
    h = mf.HAC(datos, n_clusters=4, metodo=metodo)
    plt.sca(ax)
    h.plot_dendrograma(
        max_hojas=25,
        titulo=f'{metodo.capitalize()} — cofenética: {h.cophenet_corr:.3f}'
    )

plt.tight_layout()
plt.show()
```

---

## Consejos para no estadísticos

- **Empieza con `n_clusters=3`** y observa el dendrograma para decidir si necesitas más o menos grupos.
- **El mapa de calor** es tu mejor aliado para entender *qué significa* cada cluster en términos de negocio.
- **Si un cluster tiene muy pocas observaciones** (<5% del total), considera reducir el número de clusters.
- **HAC vs K-Means:** HAC siempre da el mismo resultado con los mismos datos (es determinista). K-Means puede variar entre ejecuciones.
- Los datos se **estandarizan automáticamente** dentro de la clase — no necesitas hacerlo antes.
