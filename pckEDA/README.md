# pckEDA — Adding a New Algorithm

## Package Structure

```
pckEDA/
├── __init__.py
├── eda.py
├── no_supervisado/
│   ├── __init__.py
│   ├── base.py
│   ├── clustering.py
│   └── acp.py
└── supervisado/
    ├── __init__.py
    ├── base.py
    ├── clasificacion.py
    └── regresion.py
```

---

## Two Steps to Add a New Algorithm

### Step 1 — Create the file

Create a new `.py` file inside the correct subfolder (`no_supervisado/` or `supervisado/`),
inherit from the corresponding base class, and implement your algorithm.

**Example:** adding `KMeans` under `no_supervisado/`

```python
# no_supervisado/kmeans.py

from .base import NoSupervisado

class KMeans(NoSupervisado):
    def __init__(self, df, n_clusters):
        super().__init__(df)
        self.n_clusters = n_clusters
```

---

### Step 2 — Register it in `__init__.py`

Add the import to **both** the subpackage `__init__.py` and the root `__init__.py`.

**`no_supervisado/__init__.py`**
```python
from .base import NoSupervisado
from .clustering import Clustering
from .acp import ACP
from .kmeans import KMeans          # ← add this line

__all__ = ["NoSupervisado", "Clustering", "ACP", "KMeans"]   # ← add to __all__
```

**`pckEDA/__init__.py`**
```python
from .no_supervisado import NoSupervisado, Clustering, ACP, KMeans   # ← add here

__all__ = [
    ...
    "KMeans",    # ← add here
]
```

That's it. The new algorithm is immediately available as `mf.KMeans(...)`.

## Mandate

This package tries to follow PEPs to guarantee maintainability and standardization:

[PEP 20 – The Zen of Python](https://peps.python.org/pep-0020/)

[PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)

[PEP 257 – Docstring Conventions](https://peps.python.org/pep-0257/)

[PEP 328 – Imports: Multi-Line and Absolute/Relative](https://peps.python.org/pep-0328/)