# UMAP — Uniform Manifold Approximation and Projection

## ¿Qué hace?

**UMAP** proyecta datos de alta dimensionalidad a 2D o 3D preservando tanto la estructura
**local** (vecindades) como la **global** (relaciones entre grupos distantes). Es más rápido
que t-SNE y, a diferencia de este, permite proyectar nuevos datos una vez ajustado el modelo.

**Analogía:** Si t-SNE es como hacer un retrato de un grupo de personas (captura muy bien
los detalles locales pero distorsiona las distancias), UMAP es como hacer una foto satelital:
preserva tanto los detalles locales como la posición relativa de los grupos entre sí.

---

## ¿Cuándo usarlo?

- Cuando necesitas una proyección **más rápida** que t-SNE.
- Cuando quieres preservar tanto la **estructura local como la global**.
- Cuando necesitas proyectar **nuevos datos** sin reajustar el modelo (`transformar()`).
- Como exploración visual antes de aplicar clustering.

> **Requisito:** El paquete `umap-learn` debe estar instalado en el entorno (disponible en Anaconda).

---

## Flujo de uso

```
mf.UMAP(path, num)          # 1. Cargar datos desde CSV
    ↓ métodos EDA heredados  # 2. Limpiar y preparar
umap.ajustar()              # 3. Calcular proyección
umap.plot_proyeccion()      # 4. Visualizar
umap.transformar(nuevos)    # 5. (Opcional) Proyectar nuevos datos
```

## Creación del modelo

```python
import pckEDA as mf

umap = mf.UMAP('hotel_bookings.csv', 1, n_componentes=2, n_vecinos=15, distancia_minima=0.1)
```

### Parámetros del constructor

| Parámetro | Tipo | Valor por defecto | Descripción sencilla |
|-----------|------|-------------------|----------------------|
| `path` | `str` | — | Ruta al archivo CSV |
| `num` | `int` | — | Formato del CSV: **1** = coma con índice, **2** = punto y coma sin índice |
| `n_componentes` | `int` | `2` | Dimensiones de la proyección: `2` (gráfico 2D) o `3` (gráfico 3D) |
| `n_vecinos` | `int` | `15` | Vecinos cercanos para construir el grafo local (ver tabla abajo) |
| `distancia_minima` | `float` | `0.1` | Distancia mínima entre puntos en la proyección (0.0 – 1.0) |
| `metrica` | `str` | `'euclidean'` | Métrica de distancia para el grafo de vecinos |
| `random_state` | `int` | `42` | Semilla para reproducibilidad |

#### Guía para elegir `n_vecinos`

| Valor | Efecto |
|-------|--------|
| 5 – 15 | Resalta agrupaciones **locales** densas — ideal para detectar clusters pequeños |
| 15 – 50 | **Balance** entre estructura local y global — recomendado para la mayoría de casos |
| 50 – 200 | Preserva mejor la **estructura global** — útil para datasets con gradientes continuos |

#### Guía para elegir `distancia_minima`

| Valor | Efecto |
|-------|--------|
| 0.0 – 0.1 | Clusters **compactos** y bien separados — bueno para clustering |
| 0.1 – 0.5 | Balance entre compacidad y dispersión |
| 0.5 – 1.0 | Proyección **continua y dispersa** — bueno para visualizar gradientes |

### Método `ajustar()`

Calcula la proyección UMAP sobre `self.df`. Debe llamarse después de limpiar y
preparar los datos con los métodos heredados del EDA.

### Método `transformar(nuevos_datos)`

Proyecta observaciones nuevas al espacio UMAP ya aprendido sin reajustar el modelo.

```python
nuevas_coords = umap.transformar(nuevos_datos_escalados)
```

---

## Propiedades

| Propiedad | Qué contiene |
|-----------|--------------|
| `umap.coordenadas` | Array (n_samples × n_componentes) con las coordenadas proyectadas |
| `umap.modelo` | Modelo UMAP ajustado (permite llamar `transformar()`) |
| `umap.datos_escalados` | Datos normalizados usados internamente |
| `umap.n_vecinos` | Número de vecinos configurado |
| `umap.distancia_minima` | Distancia mínima configurada |
| `umap.metrica` | Métrica de distancia utilizada |

---

## Métodos de visualización

### `umap.plot_proyeccion(etiquetas=None, titulo='...')`

Proyección 2D. Si pasas `etiquetas` (e.g. `km.etiquetas` o `hac.etiquetas`),
cada punto se colorea según su cluster.

### `umap.plot_vecinos(valores=(5,15,30,50), titulo='...')`

Cuadrícula comparando la proyección con distintos valores de `n_vecinos`.
Útil para calibrar el parámetro antes del análisis final.

---

## Ejemplo completo

```python
import matplotlib.pyplot as plt
import pckEDA as mf

# 1. Cargar y preparar
umap = mf.UMAP('hotel_bookings.csv', 1, n_vecinos=15, distancia_minima=0.1)
umap.codificarCategorica('hotel')
umap.codificarCategorica('deposit_type', mapeo={'No Deposit': 0, 'Non Refund': 1, 'Refundable': 2})
umap.eliminarNulos()
umap.analisisNumerico()

# 2. Ajustar
umap.ajustar()

# 3. Visualizar sin etiquetas
plt.figure(figsize=(10, 8))
umap.plot_proyeccion(titulo='UMAP — Hotel Bookings')
plt.show()

# 4. Colorear con clusters de HAC
hac = mf.HAC('hotel_bookings.csv', 1, n_clusters=4)
# ... (mismo preprocesamiento) ...
hac.ajustar()

plt.figure(figsize=(10, 8))
umap.plot_proyeccion(etiquetas=hac.etiquetas, titulo='UMAP coloreado por HAC')
plt.show()

# 5. Explorar n_vecinos
umap.plot_vecinos(valores=(5, 15, 30, 50))
plt.show()
```

---

## UMAP vs t-SNE — ¿cuál elegir?

| Criterio | t-SNE | UMAP |
|----------|-------|------|
| Velocidad | Lento | Rápido |
| Estructura global preservada | Débil | Buena |
| Proyectar nuevos datos | No | Sí |
| Estabilidad del resultado | Baja | Alta |
| Parámetro más sensible | Perplejidad | n_vecinos |
| **Recomendación** | Exploración rápida | Análisis definitivo |

---

## Consejos para no estadísticos

- **Empieza con los valores por defecto** (`n_vecinos=15`, `distancia_minima=0.1`) y ajusta si el resultado no es claro.
- **`plot_vecinos()`** es tu herramienta de calibración: prueba 4 valores antes de interpretar.
- **Colorea siempre con etiquetas de clustering** — la proyección sola no asigna grupos, solo los revela visualmente.
- A diferencia de t-SNE, puedes confiar más en las **distancias relativas entre clusters** en la proyección UMAP.
