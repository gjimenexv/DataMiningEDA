"""Semana 2 — Árbol de decisión (CART)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

from ..base import ModeloBase


class ArbolDecision(ModeloBase):
    """Árbol de decisión CART para clasificación.

    Divide los datos con preguntas que minimizan la impureza (Gini o
    entropía). Es el modelo más interpretable del curso, pero sin poda
    tiene alta varianza (sobreajusta).
    """

    nombre = "Árbol de decisión"

    def __init__(self, df, target, criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, ccp_alpha=0.0,
                 class_weight=None, random_state=42):
        """Inicializa el árbol.

        Args:
            df: DataFrame con predictoras numéricas y el target.
            target: Nombre de la columna objetivo.
            criterion: ``'gini'``, ``'entropy'`` o ``'log_loss'``.
            max_depth: Profundidad máxima. ``None`` = sin límite (sobreajusta).
            min_samples_split: Mínimo de muestras para dividir un nodo.
            min_samples_leaf: Mínimo de muestras en cada hoja.
            ccp_alpha: Poda por costo-complejidad; mayor = árbol más simple.
            class_weight: ``'balanced'`` para clases desbalanceadas.
            random_state: Semilla.
        """
        super().__init__(df, target, None, random_state, criterion=criterion,
                         max_depth=max_depth, min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha,
                         class_weight=class_weight)

    def _crear_modelo(self):
        """Crea el ``DecisionTreeClassifier`` con los hiperparámetros actuales."""
        return DecisionTreeClassifier(random_state=self._random_state, **self._hp)

    @staticmethod
    def gini(y):
        """Calcula el índice de Gini de un conjunto de etiquetas.

        ``G = 1 - sum(p_i^2)``. 0 = nodo puro; 0.5 = máxima impureza (2 clases).

        Args:
            y: Secuencia de etiquetas (lista, array o Series).

        Returns:
            float: Impureza de Gini.
        """
        p = pd.Series(y).value_counts(normalize=True).to_numpy()
        return float(1 - np.sum(p ** 2))

    @classmethod
    def gini_ponderado(cls, y_izq, y_der):
        """Gini ponderado de una partición: ``N_L/N * G(L) + N_R/N * G(R)``.

        Es el valor que CART minimiza al elegir cada división.

        Args:
            y_izq: Etiquetas de la rama izquierda.
            y_der: Etiquetas de la rama derecha.

        Returns:
            float: Gini ponderado de la división.
        """
        n_i, n_d = len(y_izq), len(y_der)
        return (n_i * cls.gini(y_izq) + n_d * cls.gini(y_der)) / (n_i + n_d)

    def mejor_umbral(self, variable):
        """Busca el mejor corte de una variable numérica por Gini ponderado.

        Prueba los puntos medios entre valores consecutivos ordenados,
        ``t_i = (x_i + x_{i+1}) / 2``, como en el ejemplo de clase.

        Args:
            variable: Nombre de la columna numérica.

        Returns:
            tuple: ``(umbral, gini_ponderado)`` del mejor corte.
        """
        x, y = self._df[variable].to_numpy(), self._df[self._target].to_numpy()
        valores = np.unique(x)
        candidatos = (valores[:-1] + valores[1:]) / 2
        costos = [self.gini_ponderado(y[x <= t], y[x > t]) for t in candidatos]
        i = int(np.argmin(costos))
        return float(candidatos[i]), float(costos[i])

    def reglas_texto(self):
        """Devuelve las reglas del árbol ajustado como texto legible.

        Returns:
            str: Una línea por nodo, con la condición y la clase de las hojas.
        """
        self._verificar_ajustado()
        return export_text(self.estimador, feature_names=list(self._X_train.columns))

    def ruta_poda(self):
        """Calcula los valores de ``ccp_alpha`` disponibles para podar.

        Returns:
            DataFrame con cada ``ccp_alpha`` y la impureza total del árbol
            podado; sirve para elegir un alpha por validación cruzada.
        """
        self._verificar_datos()
        arbol = DecisionTreeClassifier(random_state=self._random_state)
        ruta = arbol.cost_complexity_pruning_path(self._X_train, self._y_train)
        return pd.DataFrame({"ccp_alpha": ruta.ccp_alphas, "impureza": ruta.impurities})

    def plot_arbol(self, max_depth=3, ax=None):
        """Dibuja el árbol ajustado.

        Args:
            max_depth: Niveles a mostrar (los árboles profundos son ilegibles).
            ax: Axes de matplotlib opcional.
        """
        self._verificar_ajustado()
        if ax is None:
            _, ax = plt.subplots(figsize=(16, 8))
        plot_tree(self.estimador, feature_names=list(self._X_train.columns),
                  class_names=[str(c) for c in self.clases], filled=True,
                  rounded=True, max_depth=max_depth, fontsize=8, ax=ax)
        ax.set_title("Árbol de decisión")
