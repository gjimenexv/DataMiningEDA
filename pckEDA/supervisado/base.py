from pckEDA.eda import AnalisisDatosExploratorio

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class Supervisado(AnalisisDatosExploratorio):
    """Clase base para algoritmos de aprendizaje supervisado.

    Hereda de AnalisisDatosExploratorio y provee la funcionalidad común a
    clasificación y regresión: variable objetivo, separación train/test,
    escalado y métricas de evaluación.

    Adaptada a partir de GuiaClaseSupervisada.py del profesor Juan Murillo Morera.
    """

    def __init__(self, df, target='target'):
        """Inicializa la clase con un DataFrame y la variable objetivo.

        Args:
            df: DataFrame de pandas con los datos a analizar.
            target: Nombre de la columna que contiene la variable objetivo.
                    Por defecto 'target'.
        """
        self.df = df
        self.__target = target
        self.__X = None
        self.__y = None
        self.__X_train = None
        self.__X_test = None
        self.__y_train = None
        self.__y_test = None
        self.__escalador = None
        self.__X_train_escalado = None
        self.__X_test_escalado = None

    # ─── Propiedades ────────────────────────────────────────────────────────

    @property
    def target(self):
        """Nombre de la columna que contiene la variable objetivo."""
        return self.__target

    @target.setter
    def target(self, target):
        """Args:
            target: Nuevo nombre de la columna objetivo.
        """
        self.__target = target

    @property
    def X(self):
        """DataFrame con las variables predictoras (features)."""
        return self.__X

    @X.setter
    def X(self, X):
        """Args:
            X: Nuevo DataFrame de features.
        """
        self.__X = X

    @property
    def y(self):
        """Series con la variable objetivo."""
        return self.__y

    @y.setter
    def y(self, y):
        """Args:
            y: Nueva Series objetivo.
        """
        self.__y = y

    @property
    def X_train(self):
        """Features de entrenamiento (sin escalar)."""
        return self.__X_train

    @X_train.setter
    def X_train(self, X_train):
        """Args:
            X_train: Nuevas features de entrenamiento.
        """
        self.__X_train = X_train

    @property
    def X_test(self):
        """Features de prueba (sin escalar)."""
        return self.__X_test

    @X_test.setter
    def X_test(self, X_test):
        """Args:
            X_test: Nuevas features de prueba.
        """
        self.__X_test = X_test

    @property
    def y_train(self):
        """Etiquetas de entrenamiento."""
        return self.__y_train

    @y_train.setter
    def y_train(self, y_train):
        """Args:
            y_train: Nuevas etiquetas de entrenamiento.
        """
        self.__y_train = y_train

    @property
    def y_test(self):
        """Etiquetas de prueba."""
        return self.__y_test

    @y_test.setter
    def y_test(self, y_test):
        """Args:
            y_test: Nuevas etiquetas de prueba.
        """
        self.__y_test = y_test

    @property
    def escalador(self):
        """Objeto escalador ajustado (StandardScaler)."""
        return self.__escalador

    @escalador.setter
    def escalador(self, escalador):
        """Args:
            escalador: Nuevo escalador.
        """
        self.__escalador = escalador

    @property
    def X_train_escalado(self):
        """Features de entrenamiento escaladas."""
        return self.__X_train_escalado

    @X_train_escalado.setter
    def X_train_escalado(self, X_train_escalado):
        """Args:
            X_train_escalado: Nuevas features escaladas de entrenamiento.
        """
        self.__X_train_escalado = X_train_escalado

    @property
    def X_test_escalado(self):
        """Features de prueba escaladas."""
        return self.__X_test_escalado

    @X_test_escalado.setter
    def X_test_escalado(self, X_test_escalado):
        """Args:
            X_test_escalado: Nuevas features escaladas de prueba.
        """
        self.__X_test_escalado = X_test_escalado

    # ─── Preparación de datos ──────────────────────────────────────────────

    def preparar_datos(self, test_size=0.25, random_state=None):
        """Separa features y target, escala con StandardScaler y divide en train/test.

        El escalador se ajusta exclusivamente sobre el conjunto de entrenamiento
        para evitar data leakage.

        *** Método adaptado de GuiaClaseSupervisada.py — Prof. Juan Murillo Morera ***

        Args:
            test_size: Proporción del conjunto de prueba. Por defecto 0.25.
            random_state: Semilla para reproducibilidad. Por defecto None.
        """
        self.__X = self.df.drop(columns=[self.__target])
        self.__y = self.df[self.__target]

        self.__X_train, self.__X_test, self.__y_train, self.__y_test = \
            train_test_split(self.__X, self.__y, test_size=test_size,
                             random_state=random_state)

        self.__escalador = StandardScaler()
        self.__X_train_escalado = pd.DataFrame(
            self.__escalador.fit_transform(self.__X_train),
            columns=self.__X_train.columns,
            index=self.__X_train.index,
        )
        self.__X_test_escalado = pd.DataFrame(
            self.__escalador.transform(self.__X_test),
            columns=self.__X_test.columns,
            index=self.__X_test.index,
        )

    # ─── Evaluación ────────────────────────────────────────────────────────

    @staticmethod
    def indices_general(MC, nombres=None):
        """Calcula índices generales de rendimiento a partir de una matriz de confusión.

        *** Método adaptado de GuiaClaseSupervisada.py — Prof. Juan Murillo Morera ***

        Args:
            MC: Matriz de confusión (numpy array).
            nombres: Lista opcional con los nombres de las categorías.

        Returns:
            dict: Diccionario con la matriz de confusión, precisión global,
                  error global y precisión por categoría.
        """
        precision_global = np.sum(MC.diagonal()) / np.sum(MC)
        error_global = 1 - precision_global
        precision_categoria = pd.DataFrame(
            MC.diagonal() / np.sum(MC, axis=1)
        ).T
        if nombres is not None:
            precision_categoria.columns = nombres
        return {
            "Matriz de Confusión": MC,
            "Precisión Global": precision_global,
            "Error Global": error_global,
            "Precisión por categoría": precision_categoria,
        }

    def evaluar(self, y_test, y_pred):
        """Imprime las métricas de evaluación a partir de predicciones.

        *** Método adaptado de GuiaClaseSupervisada.py — Prof. Juan Murillo Morera ***

        Args:
            y_test: Etiquetas reales del conjunto de prueba.
            y_pred: Predicciones del modelo.
        """
        MC = confusion_matrix(y_test, y_pred)
        indices = self.indices_general(MC, list(np.unique(self.__y)))
        for k in indices:
            print("\n%s:\n%s" % (k, str(indices[k])))
