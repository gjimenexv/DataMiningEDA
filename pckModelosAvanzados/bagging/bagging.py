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
        self._max_depth = max_depth
        super().__init__(df, target, None, random_state, n_estimators=n_estimators,
                         max_samples=max_samples, oob_score=oob_score)

    def _crear_modelo(self):
        """Crea el ``BaggingClassifier`` con árboles como estimador base."""
        base = DecisionTreeClassifier(max_depth=self._max_depth,
                                      random_state=self._random_state)
        return BaggingClassifier(estimator=base, n_jobs=-1,
                                 random_state=self._random_state, **self._hp)

    def _aplicar_params(self, params):
        """Separa la profundidad del árbol base (``estimator__max_depth``)."""
        params = dict(params)
        for clave in ("estimator__max_depth", "max_depth"):
            if clave in params:
                self._max_depth = params.pop(clave)
        super()._aplicar_params(params)

    @property
    def oob_score(self):
        """Exactitud estimada con las muestras out-of-bag del modelo ajustado."""
        self._verificar_ajustado()
        return getattr(self.estimador, "oob_score_", None)
