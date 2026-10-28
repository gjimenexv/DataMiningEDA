"""Semana 5 — XGBoost."""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from ..base import ModeloBase


class XGBoost(ModeloBase):
    """XGBoost para clasificación.

    Gradient boosting con regularización (``reg_lambda``, ``gamma``), manejo
    automático de valores faltantes y entrenamiento rápido. Requiere
    ``pip install xgboost``. Las etiquetas del target deben ser 0..K-1.
    """

    nombre = "XGBoost"
    _HIPERPARAMETROS = ("n_estimators", "learning_rate", "max_depth", "reg_lambda", "gamma", "subsample", "scale_pos_weight", "early_stopping_rounds",)

    def __init__(self, df, target, n_estimators=300, learning_rate=0.1,
                 max_depth=4, reg_lambda=1.0, gamma=0.0, subsample=1.0,
                 scale_pos_weight=None, early_stopping_rounds=None,
                 random_state=42):
        """Inicializa el XGBoost.

        Args:
            df: DataFrame con predictoras numéricas y el target (0..K-1).
            target: Nombre de la columna objetivo.
            n_estimators: Máximo de árboles (use muchos + early stopping).
            learning_rate: Tamaño de cada corrección (0.01–0.1).
            max_depth: Profundidad por árbol (3–6).
            reg_lambda: Regularización L2; desconfía de hojas con pocos datos.
            gamma: Ganancia mínima para aceptar un corte (poda).
            subsample: Fracción de filas por árbol (0.7–1).
            scale_pos_weight: Peso de la clase positiva para desbalance. Use
                ``'auto'`` para calcular n.º negativos / n.º positivos.
            early_stopping_rounds: Detiene el entrenamiento si la pérdida de
                validación no mejora en ese número de árboles. Reserva un 20 %
                del train como validación.
            random_state: Semilla.
        """
        super().__init__(df, target, None, random_state)
        self.__n_estimators = n_estimators
        self.__learning_rate = learning_rate
        self.__max_depth = max_depth
        self.__reg_lambda = reg_lambda
        self.__gamma = gamma
        self.__subsample = subsample
        self.__scale_pos_weight = scale_pos_weight
        self.__early_stopping_rounds = early_stopping_rounds

    # ─── Hiperparámetros ────────────────────────────────────────────────────
    @property
    def n_estimators(self):
        """Máximo de árboles."""
        return self.__n_estimators

    @n_estimators.setter
    def n_estimators(self, n_estimators):
        """Args:
            n_estimators: Nuevo valor.
        """
        self.__n_estimators = n_estimators

    @property
    def learning_rate(self):
        """Tamaño de cada corrección (0.01–0.1)."""
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        """Args:
            learning_rate: Nuevo valor.
        """
        self.__learning_rate = learning_rate

    @property
    def max_depth(self):
        """Profundidad por árbol."""
        return self.__max_depth

    @max_depth.setter
    def max_depth(self, max_depth):
        """Args:
            max_depth: Nuevo valor.
        """
        self.__max_depth = max_depth

    @property
    def reg_lambda(self):
        """Regularización L2 sobre las hojas."""
        return self.__reg_lambda

    @reg_lambda.setter
    def reg_lambda(self, reg_lambda):
        """Args:
            reg_lambda: Nuevo valor.
        """
        self.__reg_lambda = reg_lambda

    @property
    def gamma(self):
        """Ganancia mínima para aceptar un corte."""
        return self.__gamma

    @gamma.setter
    def gamma(self, gamma):
        """Args:
            gamma: Nuevo valor.
        """
        self.__gamma = gamma

    @property
    def subsample(self):
        """Fracción de filas por árbol."""
        return self.__subsample

    @subsample.setter
    def subsample(self, subsample):
        """Args:
            subsample: Nuevo valor.
        """
        self.__subsample = subsample

    @property
    def scale_pos_weight(self):
        """Peso de la clase positiva (``'auto'`` = negativos / positivos)."""
        return self.__scale_pos_weight

    @scale_pos_weight.setter
    def scale_pos_weight(self, scale_pos_weight):
        """Args:
            scale_pos_weight: Nuevo valor.
        """
        self.__scale_pos_weight = scale_pos_weight

    @property
    def early_stopping_rounds(self):
        """Árboles sin mejora para detener el entrenamiento (``None`` = desactivado)."""
        return self.__early_stopping_rounds

    @early_stopping_rounds.setter
    def early_stopping_rounds(self, early_stopping_rounds):
        """Args:
            early_stopping_rounds: Nuevo valor.
        """
        self.__early_stopping_rounds = early_stopping_rounds

    def _crear_modelo(self):
        """Crea el ``XGBClassifier`` con los hiperparámetros actuales.

        Raises:
            ImportError: Si ``xgboost`` no está instalado.
        """
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError("XGBoost requiere: pip install xgboost") from e
        hp = self.hiperparametros
        hp.pop("early_stopping_rounds")
        if hp["scale_pos_weight"] == "auto":
            hp["scale_pos_weight"] = (self.y_train == self.clases[0]).sum() / \
                                     (self.y_train == self.clases[1]).sum()
        if hp["scale_pos_weight"] is None:
            hp.pop("scale_pos_weight")
        return XGBClassifier(eval_metric="logloss", n_jobs=-1,
                             random_state=self.random_state, **hp)

    def _entrenar(self):
        """Entrena el modelo; con early stopping usa un 20 % del train de validación.

        Returns:
            ``Pipeline`` ajustado.
        """
        pipe = self._crear_pipeline()
        if self.early_stopping_rounds is None:
            return pipe.fit(self.X_train, self.y_train)
        X_fit, X_val, y_fit, y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2,
            random_state=self.random_state, stratify=self.y_train)
        pipe.set_params(modelo__early_stopping_rounds=self.early_stopping_rounds)
        return pipe.fit(X_fit, y_fit, modelo__eval_set=[(X_fit, y_fit), (X_val, y_val)],
                        modelo__verbose=False)

    @property
    def arboles_usados(self):
        """Número de árboles efectivamente usados (menor que ``n_estimators`` con early stopping)."""
        self._verificar_ajustado()
        mejor = getattr(self.estimador, "best_iteration", None)
        return int(mejor) + 1 if mejor is not None else self.n_estimators

    def plot_curva_aprendizaje(self, ax=None):
        """Grafica el log-loss de entrenamiento y validación por árbol.

        Requiere ``early_stopping_rounds`` (que provee el conjunto de validación).

        Args:
            ax: Axes de matplotlib opcional.
        """
        self._verificar_ajustado()
        if self.early_stopping_rounds is None:
            raise RuntimeError("Cree el modelo con early_stopping_rounds para ver la curva.")
        res = self.estimador.evals_result()
        ax = ax or plt.gca()
        ax.plot(res["validation_0"]["logloss"], label="train")
        ax.plot(res["validation_1"]["logloss"], label="validación")
        ax.axvline(self.arboles_usados - 1, color="gray", ls="--", label="mejor iteración")
        ax.set_xlabel("Árboles")
        ax.set_ylabel("Log-loss")
        ax.set_title("XGBoost — curva de aprendizaje")
        ax.legend()
