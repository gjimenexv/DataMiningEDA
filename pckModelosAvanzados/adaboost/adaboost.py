"""Semana 4 — AdaBoost."""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from ..base import ModeloBase


class AdaBoost(ModeloBase):
    """AdaBoost (AdaBoost.M1 / SAMME) para clasificación.

    Entrena clasificadores débiles en secuencia; cada uno da más peso a las
    observaciones que el anterior clasificó mal, y la predicción final es un
    voto ponderado. Muy sensible al ruido y a los outliers: use datos limpios.
    """

    nombre = "AdaBoost"

    def __init__(self, df, target, n_estimators=50, learning_rate=1.0,
                 profundidad_base=1, random_state=42):
        """Inicializa el AdaBoost.

        Args:
            df: DataFrame con predictoras numéricas y el target.
            target: Nombre de la columna objetivo.
            n_estimators: Número de clasificadores débiles. Por defecto 50.
            learning_rate: Contracción del aporte de cada clasificador.
                Valores bajos exigen más ``n_estimators``.
            profundidad_base: Profundidad de cada árbol débil. 1 = *stump*
                (el valor clásico).
            random_state: Semilla.
        """
        self._profundidad_base = profundidad_base
        super().__init__(df, target, None, random_state,
                         n_estimators=n_estimators, learning_rate=learning_rate)

    def _crear_modelo(self):
        """Crea el ``AdaBoostClassifier`` con árboles de profundidad baja."""
        base = DecisionTreeClassifier(max_depth=self._profundidad_base)
        return AdaBoostClassifier(estimator=base, random_state=self._random_state,
                                  **self._hp)

    def _aplicar_params(self, params):
        """Separa la profundidad del árbol base (``estimator__max_depth``)."""
        params = dict(params)
        for clave in ("estimator__max_depth", "profundidad_base"):
            if clave in params:
                self._profundidad_base = params.pop(clave)
        super()._aplicar_params(params)

    def pesos_clasificadores(self):
        """Peso (``alpha``) y error ponderado de cada clasificador débil.

        Returns:
            DataFrame con una fila por iteración. Un error bajo da un peso
            alto al voto de ese clasificador.
        """
        self._verificar_ajustado()
        est = self.estimador
        n = len(est.estimator_errors_)
        return pd.DataFrame({
            "error": est.estimator_errors_,
            "peso": est.estimator_weights_[:n],
        }).rename_axis("iteracion")

    def curva_iteraciones(self):
        """Exactitud de train y test acumulando los clasificadores uno a uno.

        Returns:
            DataFrame indexado por n.º de clasificadores con ``train`` y ``test``.
        """
        self._verificar_ajustado()
        est = self.estimador
        train = [(p == self._y_train).mean() for p in est.staged_predict(self._X_train)]
        test = [(p == self._y_test).mean() for p in est.staged_predict(self._X_test)]
        return pd.DataFrame({"train": train, "test": test},
                            index=range(1, len(train) + 1)).rename_axis("n_estimators")

    def plot_curva_iteraciones(self, ax=None):
        """Grafica la exactitud contra el número de clasificadores.

        Si ``test`` se estanca o baja mientras ``train`` sube, hay sobreajuste.

        Args:
            ax: Axes de matplotlib opcional.
        """
        curva = self.curva_iteraciones()
        ax = ax or plt.gca()
        curva.plot(ax=ax)
        ax.set_ylabel("Exactitud")
        ax.set_title("AdaBoost — exactitud por iteración")
