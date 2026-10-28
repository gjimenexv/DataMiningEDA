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
        super().__init__(df, target, None, random_state, n_estimators=n_estimators,
                         max_depth=max_depth, max_features=max_features,
                         min_samples_leaf=min_samples_leaf,
                         class_weight=class_weight, oob_score=oob_score)

    def _crear_modelo(self):
        """Crea el ``RandomForestClassifier`` con los hiperparámetros actuales."""
        return RandomForestClassifier(n_jobs=-1, random_state=self._random_state,
                                      **self._hp)

    @property
    def oob_score(self):
        """Exactitud out-of-bag del bosque ajustado (``None`` si está desactivada)."""
        self._verificar_ajustado()
        return getattr(self.estimador, "oob_score_", None)
