# pckEDA — Minería de Datos en Python

Un paquete modular para análisis exploratorio, reducción de dimensionalidad y clustering, diseñado para ser intuitivo tanto para usuarios técnicos como no técnicos.

---

## ¿Qué incluye?

| Clase | Tipo | ¿Qué hace? | Documentación |
|-------|------|------------|---------------|
| `AnalisisDatosExploratorio` | Base | Carga, limpia, transforma y visualiza datos | [EDA.md](EDA.md) |
| `ACP` | No supervisado | Reducción de dimensionalidad (PCA) | [ACP.md](no_supervisado/ACP.md) |
| `HAC` | No supervisado | Clustering jerárquico aglomerativo | [HAC.md](no_supervisado/HAC.md) |
| `Clustering` | No supervisado | K-Means *(en desarrollo)* | [Clustering.md](no_supervisado/Clustering.md) |
| `Clasificacion` | Supervisado | Clasificación *(en desarrollo)* | [Clasificacion.md](supervisado/Clasificacion.md) |
| `Regresion` | Supervisado | Regresión *(en desarrollo)* | [Regresion.md](supervisado/Regresion.md) |

---

## Instalación

El paquete se usa directamente desde el directorio del proyecto. Agrega la ruta al path de Python antes de importar:

```python
import sys
sys.path.insert(0, '..')   # ajusta según la ubicación de tu notebook
import pckEDA as mf
```

### Dependencias

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
prince
```

---

## Inicio rápido

### 1. Análisis exploratorio

```python
import pckEDA as mf

eda = mf.AnalisisDatosExploratorio("hotel_bookings.csv", 1)
eda.eliminarNulos()
eda.eliminarDuplicados()
eda.codificarCategorica("hotel")
eda.graficarHeatmap("Hotel Bookings")
```

### 2. Análisis de Componentes Principales (ACP)

```python
import pandas as pd
import pckEDA as mf

datos = pd.read_csv("hotel_bookings.csv", index_col=0)
datos = datos.select_dtypes(include="number").dropna()

acp = mf.ACP(datos, n_componentes=5)
print(acp.var_explicada)      # % de varianza explicada por cada componente
acp.plot_circulo()             # círculo de correlación
acp.plot_plano_principal()     # posición de las observaciones
```

### 3. Clustering Jerárquico (HAC)

```python
import pandas as pd
import pckEDA as mf

datos = pd.read_csv("hotel_bookings.csv", index_col=0)
datos = datos.select_dtypes(include="number").dropna()

hac = mf.HAC(datos, n_clusters=4, metodo='ward')
print(f"Calidad del clustering: {hac.cophenet_corr:.4f}")
hac.plot_dendrograma()         # árbol jerárquico
hac.plot_mapa_calor()          # perfil de cada cluster
hac.plot_distribucion()        # tamaño de cada cluster
print(hac.resumen)             # media de variables por cluster
```

---

## Estructura del paquete

```
pckEDA/
├── README.md                  ← este archivo
├── CONTRIBUTING.md            ← guía para añadir nuevos algoritmos
├── EDA.md                     ← documentación de AnalisisDatosExploratorio
├── __init__.py
├── eda.py
├── no_supervisado/
│   ├── ACP.md
│   ├── HAC.md
│   ├── Clustering.md
│   ├── __init__.py
│   ├── base.py
│   ├── acp.py
│   ├── hac.py
│   └── clustering.py
└── supervisado/
    ├── Clasificacion.md
    ├── Regresion.md
    ├── __init__.py
    ├── base.py
    ├── clasificacion.py
    └── regresion.py
```

---

## Estándares del paquete

El código sigue los siguientes PEPs para garantizar mantenibilidad y legibilidad:

- [PEP 8](https://peps.python.org/pep-0008/) — Guía de estilo
- [PEP 20](https://peps.python.org/pep-0020/) — El Zen de Python
- [PEP 257](https://peps.python.org/pep-0257/) — Convenciones de docstrings
- [PEP 328](https://peps.python.org/pep-0328/) — Imports relativos

¿Quieres añadir un nuevo algoritmo? Consulta [CONTRIBUTING.md](CONTRIBUTING.md).
