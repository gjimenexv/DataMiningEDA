# TSNE — t-distributed Stochastic Neighbor Embedding

## ¿Qué hace?

**t-SNE** proyecta datos de alta dimensionalidad a 2D (o 3D) preservando las **vecindades
locales**: puntos que eran cercanos en el espacio original tienden a quedar juntos en la
proyección. Es una técnica no lineal, lo que le permite revelar estructuras complejas
que ACP (lineal) no puede capturar.

**Analogía:** Imagina que tienes 20 características de cada reserva de hotel. t-SNE las
"comprime" en un mapa 2D donde las reservas similares quedan agrupadas visualmente,
sin importar cuán complicada sea la forma de los grupos.

---

## ¿Cuándo usarlo?

- Para **exploración visual** de la estructura de un dataset.
- Cuando sospechas agrupaciones no lineales que ACP no revela.
- Para **validar clustering**: si los clusters de K-Means o HAC están bien separados en
  el mapa t-SNE, la solución de clustering es coherente.
- Con datasets de **tamaño mediano** — t-SNE es lento en datasets muy grandes.

> **Importante:** t-SNE **no puede proyectar nuevos datos** tras el ajuste. Para eso, usa UMAP.

---

## Flujo de uso

```
mf.TSNE(path, num)          # 1. Cargar datos desde CSV
    ↓ métodos EDA heredados  # 2. Limpiar y preparar
tsne.ajustar()              # 3. Calcular proyección
tsne.plot_proyeccion()      # 4. Visualizar
```

## Creación del modelo

```python
import pckEDA as mf

tsne = mf.TSNE('hotel_bookings.csv', 1, n_componentes=2, perplejidad=30)
```

### Parámetros del constructor

| Parámetro | Tipo | Valor por defecto | Descripción sencilla |
|-----------|------|-------------------|----------------------|
| `path` | `str` | — | Ruta al archivo CSV |
| `num` | `int` | — | Formato del CSV: **1** = coma con índice, **2** = punto y coma sin índice |
| `n_componentes` | `int` | `2` | Dimensiones de la proyección: `2` (gráfico 2D) o `3` (gráfico 3D) |
| `perplejidad` | `float` | `30` | Número aproximado de vecinos a considerar por punto (ver tabla abajo) |
| `iteraciones` | `int` | `1000` | Número de pasos de optimización — más iteraciones = mayor calidad |
| `tasa_aprendizaje` | `float\|str` | `'auto'` | Velocidad de ajuste — `'auto'` es siempre recomendado |
| `random_state` | `int` | `42` | Semilla para reproducibilidad |

#### Guía para elegir la perplejidad

| Dataset | Rango recomendado |
|---------|-------------------|
| < 500 observaciones | 5 – 15 |
| 500 – 5,000 observaciones | 15 – 30 |
| > 5,000 observaciones | 30 – 50 |

> **Regla:** La perplejidad no debe ser mayor que el número de observaciones − 1.
> Usa `plot_perplejidad()` para comparar visualmente varios valores antes de elegir.

### Método `ajustar()`

Calcula la proyección t-SNE sobre `self.df`. Debe llamarse después de codificar
categóricas, eliminar nulos y conservar solo columnas numéricas.

---

## Propiedades

| Propiedad | Qué contiene |
|-----------|--------------|
| `tsne.coordenadas` | Array (n_samples × n_componentes) con las coordenadas proyectadas |
| `tsne.modelo` | Modelo `sklearn.TSNE` ajustado |
| `tsne.datos_escalados` | Datos normalizados usados internamente |
| `tsne.perplejidad` | Valor de perplejidad utilizado |
| `tsne.iteraciones` | Número de iteraciones configurado |
| `tsne.n_componentes` | Dimensiones de la proyección |

---

## Métodos de visualización

### `tsne.plot_proyeccion(etiquetas=None, titulo='...')`

Proyección 2D. Si pasas `etiquetas` (e.g. `km.etiquetas` o `hac.etiquetas`),
cada punto se colorea según su cluster.

| Parámetro | Descripción |
|-----------|-------------|
| `etiquetas` | Array externo de etiquetas para colorear (opcional) |
| `titulo` | Título del gráfico |

### `tsne.plot_perplejidad(valores=(5,15,30,50), titulo='...')`

Cuadrícula comparando la proyección con distintos valores de perplejidad.
Úsala antes de `ajustar()` para calibrar el parámetro. Requiere que `ajustar()` ya haya sido llamado al menos una vez (para obtener `datos_escalados`).

---

## Ejemplo completo

```python
import matplotlib.pyplot as plt
import pckEDA as mf

# 1. Cargar y preparar
tsne = mf.TSNE('hotel_bookings.csv', 1, perplejidad=30)
tsne.codificarCategorica('hotel')
tsne.codificarCategorica('deposit_type', mapeo={'No Deposit': 0, 'Non Refund': 1, 'Refundable': 2})
tsne.eliminarNulos()
tsne.analisisNumerico()

# 2. Ajustar
tsne.ajustar()
print(f"Shape proyección: {tsne.coordenadas.shape}")

# 3. Visualizar sin etiquetas
plt.figure(figsize=(10, 8))
tsne.plot_proyeccion(titulo='t-SNE — Hotel Bookings')
plt.show()

# 4. Comparar con clusters de K-Means
km = mf.KMeans('hotel_bookings.csv', 1, n_clusters=4)
# ... (mismo preprocesamiento) ...
km.ajustar()

plt.figure(figsize=(10, 8))
tsne.plot_proyeccion(etiquetas=km.etiquetas, titulo='t-SNE coloreado por K-Means')
plt.show()

# 5. Explorar perplejidad
tsne.plot_perplejidad(valores=(5, 15, 30, 50))
plt.show()
```

---

## Consejos para no estadísticos

- **El mapa t-SNE no es un mapa geográfico**: las distancias absolutas entre clusters no
  son interpretables — solo importa si los puntos están juntos o separados.
- **Prueba varios valores de perplejidad** antes de interpretar resultados. Una perplejidad
  muy baja puede crear clusters artificiales; muy alta puede fusionar grupos reales.
- **Colorea con etiquetas de clustering** para validar si K-Means o HAC encontraron grupos
  geométricamente coherentes.
- Si los resultados cambian mucho entre ejecuciones, usa `random_state=42` para fijarlos.
