# pckModelosAvanzados — Cómo añadir un modelo

Cada semana del curso puede traer modelos nuevos. Añadirlos toma cuatro pasos y
siempre deja registro en el changelog.

## Estructura de un modelo

Un folder por modelo, con el nombre del modelo en minúsculas:

```
pckModelosAvanzados/
└── svm/
    ├── __init__.py     ← exporta la clase
    ├── svm.py          ← la clase (abstracción de la librería vista en clase)
    └── SVM.md          ← guía NO técnica de uso
```

## Paso 1 — Crear la clase

Herede de `ModeloBase` e implemente **solo** `_crear_modelo`. Todo lo demás (split,
métricas, CV, tuneo, umbral, gráficos) ya viene incluido.

```python
# svm/svm.py
"""Semana N — Support Vector Machine."""

from sklearn.svm import SVC

from ..base import ModeloBase


class SVM(ModeloBase):
    """Máquina de vectores de soporte para clasificación."""

    nombre = "SVM"

    def __init__(self, df, target, C=1.0, kernel="rbf", scaler="standard",
                 random_state=42):
        """Inicializa la SVM.

        Args:
            df: DataFrame con predictoras numéricas y el target.
            target: Nombre de la columna objetivo.
            C: Penalización por error. Alto = ajusta más a los datos.
            kernel: ``'linear'``, ``'rbf'``, ``'poly'``.
            scaler: ``'standard'``, ``'minmax'`` o ``None``.
            random_state: Semilla.
        """
        super().__init__(df, target, scaler, random_state, C=C, kernel=kernel)

    def _crear_modelo(self):
        """Crea el ``SVC`` con los hiperparámetros actuales."""
        return SVC(probability=True, random_state=self._random_state, **self._hp)
```

Reglas:

- Los hiperparámetros del estimador se pasan como `**kwargs` a `super().__init__`; quedan
  en `self._hp` y se usan en `_crear_modelo`.
- Si un hiperparámetro no pertenece al estimador (p. ej. la profundidad del árbol base de
  AdaBoost), sobrescriba `_aplicar_params` como en `adaboost/adaboost.py`.
- Todos los modelos deben exponer `predict_proba` (use `probability=True` si hace falta)
  para que funcionen `evaluar()` (AUC), `plot_curva_roc()` y `optimizar_umbral()`.
- Agregue solo métodos propios del modelo (gráficos o diagnósticos particulares).

## Paso 2 — Exportar la clase

`svm/__init__.py`:

```python
from .svm import SVM

__all__ = ["SVM"]
```

`pckModelosAvanzados/__init__.py`: agregue `from .svm import SVM` y `"SVM"` en `__all__`.

## Paso 3 — Documentar

- **Docstrings** (PEP 257) en español en clase y en **cada** método: qué hace, `Args`,
  `Returns`, `Raises`.
- **`SVM.md`**: guía para no estadísticos con las secciones *¿Qué hace?* (con analogía),
  *¿Cuándo usarlo?*, *Uso básico*, tabla de *Parámetros*, *Métodos propios* y *Consejos*.
  Tome como modelo cualquiera de los `.md` existentes.
- Agregue una fila a la tabla de [README.md](README.md) y al árbol de estructura.
- Si el modelo cubre un tema nuevo, actualice `Avanzada/Indice_Modelos_Mineria_Avanzada.md`.

## Paso 4 — Probar y registrar

Antes de dar por terminado, ejecute el modelo con un dataset de ejemplo:

```python
from sklearn.datasets import load_breast_cancer
import pckModelosAvanzados as pm

df = load_breast_cancer(as_frame=True).frame
m = pm.SVM(df, "target")
m.ajustar()
print(m.evaluar())
print(m.validacion_cruzada())
```

Luego:

1. Añada la entrada en [CHANGELOG.md](CHANGELOG.md) del paquete (formato *Keep a Changelog*).
2. Incremente `__version__` en `__init__.py` (SemVer: modelo nuevo = versión *minor*).
3. Añada la entrada correspondiente en el `CHANGELOG.md` de la raíz del repositorio.

## Mandato

- [PEP 20 – The Zen of Python](https://peps.python.org/pep-0020/)
- [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 257 – Docstring Conventions](https://peps.python.org/pep-0257/)
- [PEP 328 – Imports: Multi-Line and Absolute/Relative](https://peps.python.org/pep-0328/)
