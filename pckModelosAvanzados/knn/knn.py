"""Semana 1 — K-Nearest Neighbors (KNN)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from ..base import ModeloBase


class KNN(ModeloBase):
    """Clasificador K-Nearest Neighbors.

    Clasifica cada observación según la clase mayoritaria de sus ``k``
    vecinos más cercanos. Como depende de distancias, escala las variables
    por defecto (``scaler='standard'``).
    """

    nombre = "KNN"

    def __init__(self, df, target, n_neighbors=5, weights="uniform",
                 metric="minkowski", p=2, scaler="standard", random_state=42):
        """Inicializa el KNN.

        Args:
            df: DataFrame con predictoras numéricas y el target.
            target: Nombre de la columna objetivo.
            n_neighbors: Número de vecinos K. Por defecto 5.
            weights: ``'uniform'`` (todos votan igual) o ``'distance'``
                (los vecinos cercanos pesan más).
            metric: Métrica de distancia (``'minkowski'``, ``'manhattan'``,
                ``'chebyshev'``...).
            p: Parámetro de Minkowski: 1 = Manhattan, 2 = Euclidiana.
            scaler: ``'standard'``, ``'minmax'`` o ``None``. Escalar es
                obligatorio salvo que los datos ya estén en la misma escala.
            random_state: Semilla para el split.
        """
        super().__init__(df, target, scaler, random_state, n_neighbors=n_neighbors,
                         weights=weights, metric=metric, p=p)

    def _crear_modelo(self):
        """Crea el ``KNeighborsClassifier`` con los hiperparámetros actuales."""
        return KNeighborsClassifier(**self._hp)

    def _curva_k(self, k_range, cv):
        """Exactitud media por validación cruzada para cada K (solo en train)."""
        self._verificar_datos()
        folds = StratifiedKFold(cv, shuffle=True, random_state=self._random_state)
        medias = []
        for k in k_range:
            pipe = self._crear_pipeline().set_params(modelo__n_neighbors=k)
            medias.append(cross_val_score(pipe, self._X_train, self._y_train,
                                          cv=folds, n_jobs=-1).mean())
        return pd.Series(medias, index=list(k_range), name="accuracy_cv")

    def buscar_mejor_k(self, k_range=range(1, 31), cv=5):
        """Encuentra el K con mejor exactitud en validación cruzada.

        Args:
            k_range: Valores de K a evaluar. Por defecto de 1 a 30.
            cv: Folds de la validación cruzada.

        Returns:
            int: El mejor K. Queda aplicado al modelo (llame a ``ajustar()``).
        """
        curva = self._curva_k(k_range, cv)
        mejor = int(curva.idxmax())
        self.configurar(n_neighbors=mejor)
        return mejor

    def plot_curva_k(self, k_range=range(1, 31), cv=5, ax=None):
        """Grafica la exactitud en CV contra K (sesgo-varianza).

        K pequeño sobreajusta (alta varianza); K grande subajusta (alto sesgo).

        Args:
            k_range: Valores de K a evaluar.
            cv: Folds de la validación cruzada.
            ax: Axes de matplotlib opcional.
        """
        curva = self._curva_k(k_range, cv)
        ax = ax or plt.gca()
        ax.plot(curva.index, curva.values, marker="o", color="steelblue")
        ax.axvline(self._hp["n_neighbors"], color="tomato", ls="--",
                   label=f"K actual = {self._hp['n_neighbors']}")
        ax.set_xlabel("Número de vecinos (K)")
        ax.set_ylabel("Exactitud (CV)")
        ax.set_title("KNN — exactitud vs. K")
        ax.legend()

    def plot_fronteras_decision(self, var_x, var_y, resolucion=200, ax=None):
        """Dibuja la frontera de decisión del KNN en dos variables.

        Entrena un KNN auxiliar solo con ``var_x`` y ``var_y`` (el resto de
        variables se ignora) para poder visualizar la frontera en 2D.

        Args:
            var_x: Variable en el eje X.
            var_y: Variable en el eje Y.
            resolucion: Puntos por eje de la malla.
            ax: Axes de matplotlib opcional.
        """
        self._verificar_datos()
        cols = [var_x, var_y]
        pipe = self._crear_pipeline().fit(self._X_train[cols], self._y_train)
        xx, yy = np.meshgrid(
            np.linspace(self._X[var_x].min(), self._X[var_x].max(), resolucion),
            np.linspace(self._X[var_y].min(), self._X[var_y].max(), resolucion))
        malla = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=cols)
        codigos = pd.Categorical(pipe.predict(malla), categories=self.clases).codes
        ax = ax or plt.gca()
        ax.contourf(xx, yy, codigos.reshape(xx.shape), alpha=0.3, cmap="coolwarm")
        ax.scatter(self._X_train[var_x], self._X_train[var_y],
                   c=pd.Categorical(self._y_train, categories=self.clases).codes,
                   cmap="coolwarm", edgecolor="k", s=20)
        ax.set_xlabel(var_x)
        ax.set_ylabel(var_y)
        ax.set_title(f"KNN (K={self._hp['n_neighbors']}) — frontera de decisión")
