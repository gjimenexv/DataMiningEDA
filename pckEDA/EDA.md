# AnalisisDatosExploratorio — Análisis Exploratorio de Datos (EDA)

## ¿Qué hace?

Esta clase es el **punto de entrada** del paquete `pckEDA`. Carga un archivo CSV y ofrece herramientas para limpiar, transformar y visualizar los datos antes de aplicar cualquier modelo.

Piénsalo como una caja de herramientas: antes de construir algo, necesitas conocer tus materiales, limpiarlos y organizarlos.

---

## ¿Cuándo usarla?

Siempre que vayas a trabajar con un nuevo dataset. El EDA es el primer paso de cualquier proyecto de análisis o minería de datos.

---

## Creación del objeto

```python
import pckEDA as mf

eda = mf.AnalisisDatosExploratorio("mi_archivo.csv", 1)
```

### Parámetros del constructor

| Parámetro | Tipo | Descripción sencilla |
|-----------|------|----------------------|
| `path` | `str` | Ruta al archivo CSV que quieres analizar |
| `num` | `int` | Formato del CSV: **1** = separado por comas con índice, **2** = separado por punto y coma sin índice |

> **¿Cuál usar?** Si abres el CSV en Excel y las columnas se ven bien separadas con comas, usa `1`. Si el archivo usa punto y coma como separador, usa `2`.

---

## Métodos disponibles

### Exploración inicial

| Método | Qué hace |
|--------|----------|
| `analisis()` | Resumen completo: dimensiones, primeras filas y estadísticas |
| `estadisticasDescriptivas()` | Media, desviación estándar, asimetría y curtosis por columna |
| `mostrarTamaño()` | Imprime cuántas filas y columnas tiene el dataset |
| `muestraPrimerosValores(n)` | Muestra las primeras `n` filas |
| `muestraUltimosValores(n)` | Muestra las últimas `n` filas |
| `muestraTiposDeDatos()` | Muestra el tipo de dato de cada columna (número, texto, etc.) |

### Limpieza de datos

| Método | Qué hace |
|--------|----------|
| `eliminarNulos()` | Elimina filas con valores vacíos/faltantes |
| `reemplazarNulos(columna, metodo)` | Rellena vacíos con la **media** (`"mean"`) o la **mediana** (`"median"`) |
| `eliminarDuplicados()` | Elimina filas repetidas exactamente iguales |
| `detectorDeOutliers(columna)` | Muestra los valores extremos (outliers) de una columna |
| `eliminarOutliers(columna)` | Elimina los valores extremos usando el método IQR |
| `eliminarColumna(columna)` | Elimina una columna del dataset |
| `eliminarColumnas([col1, col2])` | Elimina varias columnas a la vez |
| `eliminarFilas(columna, valor)` | Elimina filas donde una columna tenga cierto valor |
| `renombrarColumna(antiguo, nuevo)` | Cambia el nombre de una columna |

### Transformación

| Método | Qué hace |
|--------|----------|
| `codificarCategorica(columna)` | Convierte texto a números (automático) |
| `codificarCategorica(columna, mapeo={...})` | Convierte texto a números con tus propias reglas |
| `analisisNumerico()` | Conserva solo las columnas con números |
| `analisisCompleto()` | Convierte texto a columnas 0/1 (one-hot encoding) |

### Visualización

| Método | Qué hace |
|--------|----------|
| `graficarHeatmap(titulo)` | Mapa de calor de correlación entre variables |
| `graficarScatter(col_x, col_y)` | Dispersión entre dos variables |
| `graficarBoxplot(columna)` | Boxplot de una columna |
| `graficarBoxplotComparativo(x, y)` | Boxplot de `y` separado por categorías de `x` |
| `graficarBoxplotTodasColumnas()` | Un boxplot por cada columna numérica |
| `graficoBoxplotOriginal()` | Cuadrícula de boxplots (3 por fila) |
| `graficarFrecuencias(titulo, x, y, columna)` | Barras con las categorías más frecuentes |
| `graficos()` | Conjunto completo: boxplot, densidad, histograma y correlación |

---

## Ejemplo de uso típico

```python
import pckEDA as mf

eda = mf.AnalisisDatosExploratorio("hotel_bookings.csv", 1)

# Explora
eda.mostrarTamaño()
eda.estadisticasDescriptivas()

# Limpia
eda.eliminarDuplicados()
eda.eliminarNulos()
eda.eliminarOutliers("adr")

# Transforma
eda.codificarCategorica("hotel")
eda.codificarCategorica("deposit_type", mapeo={"No Deposit": 0, "Non Refund": 1, "Refundable": 2})

# Visualiza
eda.graficarHeatmap("Hotel Bookings")
eda.graficarScatter("adr", "lead_time")
```

---

## Propiedad de acceso al DataFrame

```python
eda.df   # accede al DataFrame limpio en cualquier momento
```

---

## Consejos para no estadísticos

- **Siempre empieza** con `estadisticasDescriptivas()` para entender el rango de tus datos.
- **Elimina nulos antes** de cualquier análisis o modelo; los valores vacíos rompen los algoritmos.
- **Los outliers** son valores muy alejados del resto. Pueden ser errores de captura o casos genuinamente excepcionales — revísalos antes de eliminarlos.
- La **correlación** (heatmap) te dice qué variables "se mueven juntas". Un valor cercano a 1 o -1 indica relación fuerte; cercano a 0 indica independencia.
