"""Semana 5 — XGBoost."""

import pandas as pd
import matplotlib.pyplot as plt

from ..base import ModeloBase


class XGBoost(ModeloBase):
    """XGBoost para clasificación.

    Gradient boosting con regularización (``reg_lambda``, ``gamma``), manejo
    automático de valores faltantes y entrenamiento rápido. Requiere
    ``pip install xgboost``. Las etiquetas del target deben ser 0..K-1.
    """

    nombre = "XGBoost"

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
        self._early_stopping_rounds = early_stopping_rounds
        self._historial = None
        super().__init__(df, target, None, random_state, n_estimators=n_estimators,
                         learning_rate=learning_rate, max_depth=max_depth,
                         reg_lambda=reg_lambda, gamma=gamma, subsample=subsample,
                         scale_pos_weight=scale_pos_weight)

    def _crear_modelo(self):
        """Crea el ``XGBClassifier`` con los hiperparámetros actuales.

        Raises:
            ImportError: Si ``xgboost`` no está instalado.
        """
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError("XGBoost requiere: pip install xgboost") from e
        hp = dict(self._hp)
        if hp.get("scale_pos_weight") == "auto":
            hp["scale_pos_weight"] = (self._y_train == self.clases[0]).sum() / \
                                     (self._y_train == self.clases[1]).sum()
        if hp.get("scale_pos_weight") is None:
            hp.pop("scale_pos_weight")
        return XGBClassifier(eval_metric="logloss", n_jobs=-1,
                             random_state=self._random_state, **hp)

    def ajustar(self):
        """Entrena el modelo; con early stopping usa un 20 % del train de validación."""
        if self._early_stopping_rounds is None:
            super().ajustar()
            return
        from sklearn.model_selection import train_test_split
        self._verificar_datos()
        X_fit, X_val, y_fit, y_val = train_test_split(
            self._X_train, self._y_train, test_size=0.2,
            random_state=self._random_state, stratify=self._y_train)
        self._modelo = self._crear_pipeline()
        self._modelo.set_params(modelo__early_stopping_rounds=self._early_stopping_rounds)
        self._modelo.fit(X_fit, y_fit, modelo__eval_set=[(X_fit, y_fit), (X_val, y_val)],
                         modelo__verbose=False)
        self._umbral = 0.5
        self._calcular_predicciones()

    @property
    def arboles_usados(self):
        """Número de árboles efectivamente usados (menor que ``n_estimators`` con early stopping)."""
        self._verificar_ajustado()
        mejor = getattr(self.estimador, "best_iteration", None)
        return int(mejor) + 1 if mejor is not None else self._hp["n_estimators"]

    def plot_curva_aprendizaje(self, ax=None):
        """Grafica el log-loss de entrenamiento y validación por árbol.

        Requiere ``early_stopping_rounds`` (que provee el conjunto de validación).

        Args:
            ax: Axes de matplotlib opcional.
        """
        self._verificar_ajustado()
        if self._early_stopping_rounds is None:
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
