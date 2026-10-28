"""Semana 3 — Random Forest."""

from sklearn.ensemble import RandomForestClassifier

from ..base import ModeloBase


class RandomForest(ModeloBase):
    """Random Forest para clasificación.

    Promedia (vota) muchos árboles entrenados sobre muestras bootstrap y con
    un subconjunto aleatorio de variables en cada división (``max_features``).
    Reduce la varianza; el sesgo casi no cambia. No requiere escalar.
    """

    nombre = "Random Forest"
    _HIPERPARAMETROS = ("n_estimators", "max_depth", "max_features", "min_samples_leaf", "class_weight", "oob_score",)


    def __init__(self, df, target, n_estimators=100, max_depth=None,
                 max_features="sqrt", min_samples_leaf=1, class_weight=None,
                 oob_score=True, random_state=42):
        """Inicializa el Random Forest.

        Args:
            df: DataFrame con predictoras numéricas y el target.
            target: Nombre de la columna objetivo.
            n_estimators: Número de árboles (100–500). Más árboles = menos
                varianza pero más costo, con rendimientos decrecientes.
            max_depth: Profundidad máxima de cada árbol.
            max_features: Variables sorteadas en cada división (``'sqrt'`` es
                el estándar en clasificación). Decorrelaciona los árboles.
            min_samples_leaf: Mínimo de muestras por hoja.
            class_weight: ``'balanced'`` o ``'balanced_subsample'`` si hay desbalance.
            oob_score: Calcula la exactitud out-of-bag (sin set de validación).
            random_state: Semilla.
        """
        super().__init__(df, target, None, random_state)
        self.__n_estimators = n_estimators
        self.__max_depth = max_depth
        self.__max_features = max_features
        self.__min_samples_leaf = min_samples_leaf
        self.__class_weight = class_weight
        self.__oob_score = oob_score

    # ─── Hiperparámetros ────────────────────────────────────────────────────
    @property
    def n_estimators(self):
        """Número de árboles."""
        return self.__n_estimators

    @n_estimators.setter
    def n_estimators(self, n_estimators):
        """Args:
            n_estimators: Nuevo valor.
        """
        self.__n_estimators = n_estimators

    @property
    def max_depth(self):
        """Profundidad máxima de cada árbol."""
        return self.__max_depth

    @max_depth.setter
    def max_depth(self, max_depth):
        """Args:
            max_depth: Nuevo valor.
        """
        self.__max_depth = max_depth

    @property
    def max_features(self):
        """Variables sorteadas en cada división."""
        return self.__max_features

    @max_features.setter
    def max_features(self, max_features):
        """Args:
            max_features: Nuevo valor.
        """
        self.__max_features = max_features

    @property
    def min_samples_leaf(self):
        """Mínimo de muestras por hoja."""
        return self.__min_samples_leaf

    @min_samples_leaf.setter
    def min_samples_leaf(self, min_samples_leaf):
        """Args:
            min_samples_leaf: Nuevo valor.
        """
        self.__min_samples_leaf = min_samples_leaf

    @property
    def class_weight(self):
        """Pesos por clase (``'balanced'`` para desbalance)."""
        return self.__class_weight

    @class_weight.setter
    def class_weight(self, class_weight):
        """Args:
            class_weight: Nuevo valor.
        """
        self.__class_weight = class_weight

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
        """Crea el ``RandomForestClassifier`` con los hiperparámetros actuales."""
        return RandomForestClassifier(n_jobs=-1, random_state=self.random_state,
                                      **self.hiperparametros)

    @property
    def exactitud_oob(self):
        """Exactitud out-of-bag del bosque ajustado (``None`` si está desactivada)."""
        self._verificar_ajustado()
        return getattr(self.estimador, "oob_score_", None)
