"""Semana 3 — Bagging (bootstrap aggregating)."""

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from ..base import ModeloBase


class Bagging(ModeloBase):
    """Bagging de árboles de decisión.

    Entrena ``n_estimators`` árboles sobre muestras bootstrap (con
    reemplazo) y combina las predicciones por votación. Reduce la varianza
    de un árbol individual sin cambiar mucho su sesgo. Es el paso previo a
    Random Forest, que además sortea las variables en cada división.
    """

    nombre = "Bagging"
    _HIPERPARAMETROS = ("n_estimators", "max_depth", "max_samples", "oob_score",)


    def __init__(self, df, target, n_estimators=100, max_depth=None,
                 max_samples=1.0, oob_score=True, random_state=42):
        """Inicializa el Bagging.

        Args:
            df: DataFrame con predictoras numéricas y el target.
            target: Nombre de la columna objetivo.
            n_estimators: Cantidad de árboles. Por defecto 100.
            max_depth: Profundidad máxima de cada árbol base (``None`` = sin límite).
                En ``buscar_hiperparametros`` use la clave ``estimator__max_depth``.
            max_samples: Fracción de filas de cada muestra bootstrap.
            oob_score: Si es True estima el error con las filas no muestreadas
                (~37 % de los datos), sin necesitar validación aparte.
            random_state: Semilla.
        """
        super().__init__(df, target, None, random_state)
        self.__n_estimators = n_estimators
        self.__max_depth = max_depth
        self.__max_samples = max_samples
        self.__oob_score = oob_score

    # ─── Hiperparámetros ────────────────────────────────────────────────────
    @property
    def n_estimators(self):
        """Cantidad de árboles."""
        return self.__n_estimators

    @n_estimators.setter
    def n_estimators(self, n_estimators):
        """Args:
            n_estimators: Nuevo valor.
        """
        self.__n_estimators = n_estimators

    @property
    def max_depth(self):
        """Profundidad máxima de cada árbol base (``None`` = sin límite)."""
        return self.__max_depth

    @max_depth.setter
    def max_depth(self, max_depth):
        """Args:
            max_depth: Nuevo valor.
        """
        self.__max_depth = max_depth

    @property
    def max_samples(self):
        """Fracción de filas de cada muestra bootstrap."""
        return self.__max_samples

    @max_samples.setter
    def max_samples(self, max_samples):
        """Args:
            max_samples: Nuevo valor.
        """
        self.__max_samples = max_samples

    @property
    def oob_score(self):
        """Si es True calcula la exactitud out-of-bag."""
        return self.__oob_score

    @oob_score.setter
    def oob_score(self, oob_score):
        """Args:
            oob_score: Nuevo valor.
        """
        self.__oob_score = oob_score

    def _crear_modelo(self):
        """Crea el ``BaggingClassifier`` con árboles como estimador base."""
        base = DecisionTreeClassifier(max_depth=self.max_depth,
                                      random_state=self.random_state)
        hp = self.hiperparametros
        hp.pop("max_depth")
        return BaggingClassifier(estimator=base, n_jobs=-1,
                                 random_state=self.random_state, **hp)

    def _aplicar_params(self, params):
        """Traduce ``estimator__max_depth`` (clave de la búsqueda) a ``max_depth``."""
        params = {("max_depth" if k == "estimator__max_depth" else k): v
                  for k, v in params.items()}
        super()._aplicar_params(params)

    @property
    def exactitud_oob(self):
        """Exactitud estimada con las muestras out-of-bag del modelo ajustado."""
        self._verificar_ajustado()
        return getattr(self.estimador, "oob_score_", None)
