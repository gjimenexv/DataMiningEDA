"""Semana 6 — Análisis Discriminante Lineal (LDA)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ..base import ModeloBase


class LDA(ModeloBase):
    """Análisis Discriminante Lineal.

    Clasifica y reduce dimensionalidad proyectando los datos en la dirección
    que maximiza la separación entre clases y minimiza la dispersión dentro
    de cada una (criterio de Fisher). Supone normalidad en cada clase y la
    misma matriz de covarianza; da fronteras lineales.
    """

    nombre = "LDA"
    _HIPERPARAMETROS = ("solver", "shrinkage",)


    def __init__(self, df, target, solver="svd", shrinkage=None,
                 scaler="standard", random_state=42):
        """Inicializa el LDA.

        Args:
            df: DataFrame con predictoras continuas y el target.
            target: Nombre de la columna objetivo.
            solver: ``'svd'`` (por defecto), ``'lsqr'`` o ``'eigen'``.
                ``shrinkage`` solo funciona con ``'lsqr'`` o ``'eigen'``.
            shrinkage: ``None``, ``'auto'`` o un valor en [0, 1] que
                regulariza la covarianza (útil con muchas variables o pocos datos).
            scaler: ``'standard'`` (recomendado), ``'minmax'`` o ``None``.
            random_state: Semilla para el split.
        """
        super().__init__(df, target, scaler, random_state)
        self.__solver = solver
        self.__shrinkage = shrinkage

    # ─── Hiperparámetros ────────────────────────────────────────────────────
    @property
    def solver(self):
        """Método de cálculo: ``'svd'``, ``'lsqr'`` o ``'eigen'``."""
        return self.__solver

    @solver.setter
    def solver(self, solver):
        """Args:
            solver: Nuevo valor.
        """
        self.__solver = solver

    @property
    def shrinkage(self):
        """Regularización de la covarianza (``None``, ``'auto'`` o [0, 1])."""
        return self.__shrinkage

    @shrinkage.setter
    def shrinkage(self, shrinkage):
        """Args:
            shrinkage: Nuevo valor.
        """
        self.__shrinkage = shrinkage

    def _crear_modelo(self):
        """Crea el ``LinearDiscriminantAnalysis`` con los hiperparámetros actuales."""
        return LinearDiscriminantAnalysis(**self.hiperparametros)

    def transformar(self, X=None):
        """Proyecta datos sobre los ejes discriminantes.

        Hay como máximo ``K - 1`` ejes (K = número de clases).

        Args:
            X: DataFrame a proyectar. Si es None, usa el conjunto de prueba.

        Returns:
            DataFrame con columnas ``LD1``, ``LD2``, ...
        """
        self._verificar_ajustado()
        X = self.X_test if X is None else X
        Z = self.modelo.transform(X)
        return pd.DataFrame(Z, index=X.index,
                            columns=[f"LD{i + 1}" for i in range(Z.shape[1])])

    @property
    def varianza_explicada(self):
        """Proporción de separación explicada por cada eje discriminante.

        Solo disponible con ``solver`` ``'svd'`` o ``'eigen'``.
        """
        self._verificar_ajustado()
        return getattr(self.estimador, "explained_variance_ratio_", None)

    def covarianzas_por_clase(self):
        """Compara la covarianza de cada clase (supuesto de homocedasticidad).

        Si las covarianzas son muy distintas entre clases, el supuesto de LDA
        no se cumple y conviene probar QDA.

        Returns:
            DataFrame con el determinante y la traza de la covarianza de cada clase.
        """
        self._verificar_datos()
        filas = {}
        for c in self.clases:
            cov = np.cov(self.X_train[self.y_train == c].to_numpy(), rowvar=False)
            filas[c] = {"determinante": np.linalg.det(cov), "traza": np.trace(cov)}
        return pd.DataFrame(filas).T

    def plot_proyeccion(self, ax=None):
        """Grafica el conjunto de prueba sobre los ejes discriminantes.

        Con 2 clases se dibuja un histograma de LD1; con 3 o más, LD1 vs LD2.

        Args:
            ax: Axes de matplotlib opcional.
        """
        Z = self.transformar()
        ax = ax or plt.gca()
        if Z.shape[1] == 1:
            for c in self.clases:
                ax.hist(Z.loc[self.y_test == c, "LD1"], alpha=0.6, bins=25, label=str(c))
            ax.set_xlabel("LD1")
            ax.legend()
        else:
            for c in self.clases:
                m = self.y_test == c
                ax.scatter(Z.loc[m, "LD1"], Z.loc[m, "LD2"], alpha=0.7, label=str(c))
            ax.set_xlabel("LD1")
            ax.set_ylabel("LD2")
            ax.legend()
        ax.set_title("LDA — proyección discriminante")
