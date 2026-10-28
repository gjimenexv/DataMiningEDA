"""Semana 5 — Gradient Boosting."""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

from ..base import ModeloBase


class GradientBoosting(ModeloBase):
    """Gradient Boosting (scikit-learn) para clasificación.

    Parte de una predicción inicial y suma árboles pequeños, cada uno
    entrenado para predecir los residuos del anterior, escalados por la tasa
    de aprendizaje. Los árboles suman en escala de log-odds.
    """

    nombre = "Gradient Boosting"

    def __init__(self, df, target, n_estimators=100, learning_rate=0.1,
                 max_depth=3, subsample=1.0, n_iter_no_change=None,
                 random_state=42):
        """Inicializa el Gradient Boosting.

        Args:
            df: DataFrame con predictoras numéricas y el target.
            target: Nombre de la columna objetivo.
            n_estimators: Máximo de árboles.
            learning_rate: Tamaño de cada corrección (0.01–0.1 es lo típico).
            max_depth: Profundidad de cada árbol (3–6).
            subsample: Fracción de filas por árbol (<1 añade regularización).
            n_iter_no_change: Si se fija, activa *early stopping*: detiene el
                entrenamiento cuando la validación interna no mejora en ese
                número de iteraciones.
            random_state: Semilla.
        """
        super().__init__(df, target, None, random_state, n_estimators=n_estimators,
                         learning_rate=learning_rate, max_depth=max_depth,
                         subsample=subsample, n_iter_no_change=n_iter_no_change)

    def _crear_modelo(self):
        """Crea el ``GradientBoostingClassifier`` con los hiperparámetros actuales."""
        return GradientBoostingClassifier(random_state=self._random_state, **self._hp)

    def curva_iteraciones(self):
        """Exactitud de train y test acumulando los árboles uno a uno.

        Returns:
            DataFrame indexado por n.º de árboles con ``train`` y ``test``.
        """
        self._verificar_ajustado()
        est = self.estimador
        train = [(p == self._y_train).mean() for p in est.staged_predict(self._X_train)]
        test = [(p == self._y_test).mean() for p in est.staged_predict(self._X_test)]
        return pd.DataFrame({"train": train, "test": test},
                            index=range(1, len(train) + 1)).rename_axis("n_estimators")

    def plot_curva_iteraciones(self, ax=None):
        """Grafica la exactitud contra el número de árboles.

        El punto donde ``test`` deja de mejorar indica cuántos árboles usar.

        Args:
            ax: Axes de matplotlib opcional.
        """
        ax = ax or plt.gca()
        self.curva_iteraciones().plot(ax=ax)
        ax.set_ylabel("Exactitud")
        ax.set_title("Gradient Boosting — exactitud por árbol")
