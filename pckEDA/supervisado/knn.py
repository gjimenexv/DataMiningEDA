from .clasificacion import Clasificacion

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


class KNN(Clasificacion):
    """Clase para realizar clasificación K-Nearest Neighbors (KNN) en un DataFrame.

    KNN clasifica una observación según la clase mayoritaria de sus k vecinos
    más cercanos en el espacio de features. Al depender de distancias, requiere
    que las variables estén escaladas.

    Hereda de Clasificacion → Supervisado → AnalisisDatosExploratorio.
    """

    def __init__(self, df, target, n_neighbors=5, weights='uniform',
                 metric='euclidean', scaler='standard'):
        """Inicializa la clase KNN con un DataFrame y la variable objetivo.

        Args:
            df: DataFrame de pandas con los datos a analizar.
            target: Nombre de la columna que contiene la variable objetivo (clase).
            n_neighbors: Número de vecinos a considerar. Por defecto 5.
            weights: Función de pesos: 'uniform' o 'distance'. Por defecto 'uniform'.
            metric: Métrica de distancia: 'euclidean', 'manhattan', 'minkowski'.
                    Por defecto 'euclidean'.
            scaler: Tipo de escalador: 'standard' o 'minmax'. Por defecto 'standard'.
        """
        super().__init__(df, target)
        self.__n_neighbors = n_neighbors
        self.__weights = weights
        self.__metric = metric
        self.__scaler_type = scaler
        self.__X = None
        self.__y = None
        self.__X_train = None
        self.__X_test = None
        self.__y_train = None
        self.__y_test = None
        self.__escalador = None
        self.__X_train_escalado = None
        self.__X_test_escalado = None
        self.__modelo = None
        self.__predicciones = None
        self.__probabilidades = None
        self.__accuracy = None
        self.__matriz_confusion = None
        self.__reporte = None
        self.__mejores_params = None
        self.__nombres_clases = None

    # ─── Propiedades ────────────────────────────────────────────────────────

    @property
    def n_neighbors(self):
        """Devuelve el número de vecinos configurado."""
        return self.__n_neighbors

    @n_neighbors.setter
    def n_neighbors(self, n_neighbors):
        """Args:
            n_neighbors: Nuevo número de vecinos.
        """
        self.__n_neighbors = n_neighbors

    @property
    def weights(self):
        """Devuelve la función de pesos configurada."""
        return self.__weights

    @weights.setter
    def weights(self, weights):
        """Args:
            weights: Nueva función de pesos ('uniform' o 'distance').
        """
        self.__weights = weights

    @property
    def metric(self):
        """Devuelve la métrica de distancia configurada."""
        return self.__metric

    @metric.setter
    def metric(self, metric):
        """Args:
            metric: Nueva métrica de distancia.
        """
        self.__metric = metric

    @property
    def scaler_type(self):
        """Devuelve el tipo de escalador configurado."""
        return self.__scaler_type

    @scaler_type.setter
    def scaler_type(self, scaler_type):
        """Args:
            scaler_type: Nuevo tipo de escalador ('standard' o 'minmax').
        """
        self.__scaler_type = scaler_type

    @property
    def X(self):
        """Devuelve el DataFrame con las variables predictoras."""
        return self.__X

    @X.setter
    def X(self, X):
        """Args:
            X: Nuevo DataFrame de features.
        """
        self.__X = X

    @property
    def y(self):
        """Devuelve la Series con la variable objetivo."""
        return self.__y

    @y.setter
    def y(self, y):
        """Args:
            y: Nueva Series de etiquetas.
        """
        self.__y = y

    @property
    def X_train(self):
        """Devuelve las features de entrenamiento (sin escalar)."""
        return self.__X_train

    @X_train.setter
    def X_train(self, X_train):
        """Args:
            X_train: Nuevas features de entrenamiento.
        """
        self.__X_train = X_train

    @property
    def X_test(self):
        """Devuelve las features de prueba (sin escalar)."""
        return self.__X_test

    @X_test.setter
    def X_test(self, X_test):
        """Args:
            X_test: Nuevas features de prueba.
        """
        self.__X_test = X_test

    @property
    def y_train(self):
        """Devuelve las etiquetas de entrenamiento."""
        return self.__y_train

    @y_train.setter
    def y_train(self, y_train):
        """Args:
            y_train: Nuevas etiquetas de entrenamiento.
        """
        self.__y_train = y_train

    @property
    def y_test(self):
        """Devuelve las etiquetas de prueba."""
        return self.__y_test

    @y_test.setter
    def y_test(self, y_test):
        """Args:
            y_test: Nuevas etiquetas de prueba.
        """
        self.__y_test = y_test

    @property
    def escalador(self):
        """Devuelve el objeto escalador ajustado."""
        return self.__escalador

    @escalador.setter
    def escalador(self, escalador):
        """Args:
            escalador: Nuevo escalador.
        """
        self.__escalador = escalador

    @property
    def modelo(self):
        """Devuelve el modelo KNN entrenado."""
        return self.__modelo

    @modelo.setter
    def modelo(self, modelo):
        """Args:
            modelo: Nuevo modelo KNN.
        """
        self.__modelo = modelo

    @property
    def predicciones(self):
        """Devuelve las predicciones sobre el conjunto de prueba."""
        return self.__predicciones

    @predicciones.setter
    def predicciones(self, predicciones):
        """Args:
            predicciones: Nuevo array de predicciones.
        """
        self.__predicciones = predicciones

    @property
    def probabilidades(self):
        """Devuelve las probabilidades de predicción por clase."""
        return self.__probabilidades

    @probabilidades.setter
    def probabilidades(self, probabilidades):
        """Args:
            probabilidades: Nuevo array de probabilidades.
        """
        self.__probabilidades = probabilidades

    @property
    def accuracy(self):
        """Devuelve la exactitud del modelo (0-1)."""
        return self.__accuracy

    @accuracy.setter
    def accuracy(self, accuracy):
        """Args:
            accuracy: Nueva exactitud.
        """
        self.__accuracy = accuracy

    @property
    def matriz_confusion(self):
        """Devuelve la matriz de confusión como numpy array."""
        return self.__matriz_confusion

    @matriz_confusion.setter
    def matriz_confusion(self, matriz_confusion):
        """Args:
            matriz_confusion: Nueva matriz de confusión.
        """
        self.__matriz_confusion = matriz_confusion

    @property
    def reporte(self):
        """Devuelve el reporte de clasificación (precision, recall, f1)."""
        return self.__reporte

    @reporte.setter
    def reporte(self, reporte):
        """Args:
            reporte: Nuevo reporte de clasificación.
        """
        self.__reporte = reporte

    @property
    def mejores_params(self):
        """Devuelve los mejores hiperparámetros encontrados por GridSearchCV."""
        return self.__mejores_params

    @mejores_params.setter
    def mejores_params(self, mejores_params):
        """Args:
            mejores_params: Nuevos mejores parámetros.
        """
        self.__mejores_params = mejores_params

    @property
    def nombres_clases(self):
        """Devuelve los nombres de las clases de la variable objetivo."""
        return self.__nombres_clases

    @nombres_clases.setter
    def nombres_clases(self, nombres_clases):
        """Args:
            nombres_clases: Nuevos nombres de clases.
        """
        self.__nombres_clases = nombres_clases

    # ─── Preparación de datos ───────────────────────────────────────────────

    def preparar_datos(self, test_size=0.2, random_state=42):
        """Separa features y target, escala los datos y divide en train/test.

        Utiliza StandardScaler o MinMaxScaler según la configuración del
        constructor. La estratificación se aplica automáticamente para
        mantener la proporción de clases en ambos conjuntos.

        Args:
            test_size: Proporción del conjunto de prueba. Por defecto 0.2.
            random_state: Semilla para reproducibilidad. Por defecto 42.
        """
        self.__X = self._df.drop(columns=[self.target])
        self.__y = self._df[self.target]
        self.__nombres_clases = sorted(self.__y.unique().tolist())

        self.__X_train, self.__X_test, self.__y_train, self.__y_test = \
            train_test_split(self.__X, self.__y, test_size=test_size,
                             random_state=random_state, stratify=self.__y)

        if self.__scaler_type == 'minmax':
            self.__escalador = MinMaxScaler()
        else:
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

    # ─── Reducción de dimensionalidad ───────────────────────────────────────

    def aplicar_pca(self, n_componentes=2):
        """Aplica PCA sobre los datos escalados para reducir dimensionalidad.

        Útil para mitigar la maldición de la dimensionalidad en KNN.
        Debe llamarse después de preparar_datos() y antes de ajustar().

        Args:
            n_componentes: Número de componentes principales. Por defecto 2.
        """
        if self.__X_train_escalado is None:
            self.preparar_datos()

        pca = PCA(n_components=n_componentes)
        cols = [f'PC{i + 1}' for i in range(n_componentes)]

        self.__X_train_escalado = pd.DataFrame(
            pca.fit_transform(self.__X_train_escalado),
            columns=cols,
            index=self.__X_train.index,
        )
        self.__X_test_escalado = pd.DataFrame(
            pca.transform(self.__X_test_escalado),
            columns=cols,
            index=self.__X_test.index,
        )

    # ─── Entrenamiento ──────────────────────────────────────────────────────

    def ajustar(self):
        """Entrena el modelo KNN sobre los datos de entrenamiento escalados.

        Debe llamarse después de preparar_datos(). Calcula automáticamente
        las predicciones, probabilidades, exactitud, matriz de confusión
        y reporte de clasificación.
        """
        if self.__X_train_escalado is None:
            self.preparar_datos()

        self.__modelo = KNeighborsClassifier(
            n_neighbors=self.__n_neighbors,
            weights=self.__weights,
            metric=self.__metric,
        )
        self.__modelo.fit(self.__X_train_escalado, self.__y_train)

        self.__predicciones = self.__modelo.predict(self.__X_test_escalado)
        self.__probabilidades = self.__modelo.predict_proba(
            self.__X_test_escalado
        )
        self.__accuracy = accuracy_score(self.__y_test, self.__predicciones)
        self.__matriz_confusion = confusion_matrix(
            self.__y_test, self.__predicciones,
        )
        self.__reporte = classification_report(
            self.__y_test, self.__predicciones,
        )

    # ─── Búsqueda de hiperparámetros ────────────────────────────────────────

    def buscar_mejor_k(self, k_range=range(1, 31),
                       weights_options=('uniform', 'distance'), cv=5):
        """Busca el mejor k y pesos usando GridSearchCV con validación cruzada.

        Args:
            k_range: Rango de valores de k a evaluar. Por defecto range(1, 31).
            weights_options: Opciones de pesos a evaluar.
                             Por defecto ('uniform', 'distance').
            cv: Número de folds para validación cruzada. Por defecto 5.

        Returns:
            dict: Diccionario con los mejores hiperparámetros encontrados.
        """
        if self.__X_train_escalado is None:
            self.preparar_datos()

        param_grid = {
            'n_neighbors': list(k_range),
            'weights': list(weights_options),
        }
        grid = GridSearchCV(
            KNeighborsClassifier(metric=self.__metric),
            param_grid,
            cv=cv,
            scoring='accuracy',
        )
        grid.fit(self.__X_train_escalado, self.__y_train)
        self.__mejores_params = grid.best_params_
        return self.__mejores_params

    # ═══════════════════════════════════════════════════════════════════════
    # ███  MÉTODOS PROVENIENTES DE ModMC.py (Profesor Juan Murillo Morera) ███
    # ═══════════════════════════════════════════════════════════════════════

    def indices_general(self, nombres=None):
        """Calcula índices generales a partir de la matriz de confusión.

        *** Método adaptado de ModMC.py — función indices_general() ***

        Args:
            nombres: Lista con los nombres de las categorías. Por defecto None.

        Returns:
            dict: Diccionario con la matriz de confusión, precisión global,
                  error global y precisión por categoría.
        """
        MC = self.__matriz_confusion
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

    def indices_binarios(self):
        """Calcula métricas detalladas para una matriz de confusión 2x2.

        Incluye precisión positiva, negativa, proporciones de falsos positivos
        y negativos, y asertividad positiva y negativa.

        *** Método adaptado de ModMC.py — clase MatrizDeConfusion ***

        Returns:
            dict: Diccionario con PP, PN, PFP, PFN, AP, AN y métricas globales.
                  Retorna None si la matriz no es 2x2.
        """
        MC = self.__matriz_confusion
        if MC.shape != (2, 2):
            return None
        VP, FP = MC[0][0], MC[0][1]
        FN, VN = MC[1][0], MC[1][1]
        PG = (VP + VN) / (VP + VN + FP + FN)
        EG = 1 - PG
        PP = VP / (VP + FP) if (VP + FP) else 0
        PN = VN / (VN + FN) if (VN + FN) else 0
        PFP = FP / (VN + FP) if (VN + FP) else 0
        PFN = FN / (VP + FN) if (VP + FN) else 0
        AP = VP / (VP + FN) if (VP + FN) else 0
        AN = VN / (VN + FP) if (VN + FP) else 0
        return {
            "Precisión Global": PG,
            "Error Global": EG,
            "Precisión Positiva (PP)": PP,
            "Precisión Negativa (PN)": PN,
            "Proporción de Falsos Positivos (PFP)": PFP,
            "Proporción de Falsos Negativos (PFN)": PFN,
            "Asertividad Positiva (AP)": AP,
            "Asertividad Negativa (AN)": AN,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # ███  GRÁFICOS PROVENIENTES DE ModMC.py (Profesor Juan Murillo Morera) ███
    # ═══════════════════════════════════════════════════════════════════════

    def plot_distribucion_target(self, ax=None):
        """Muestra la distribución de la variable objetivo como barras apiladas.

        *** Gráfico adaptado de ModMC.py — distribucion_variable_predecir() ***

        Args:
            ax: Axes de matplotlib opcional. Si es None, crea una nueva figura.
        """
        plt.style.use('seaborn-v0_8-bright')
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(15, 10), dpi=200)
        colors = list(dict(**mcolors.CSS4_COLORS))
        data = self._df
        variable_predict = self.target
        df_dist = pd.crosstab(
            index=data[variable_predict], columns="valor"
        ) / data[variable_predict].count()

        countv = 0
        titulo = "Distribución de la variable %s" % variable_predict
        for i in range(df_dist.shape[0]):
            ax.barh(
                1, df_dist.iloc[i], left=countv, align='center',
                color=colors[11 + i], label=df_dist.iloc[i].name,
            )
            countv = countv + df_dist.iloc[i]

        ax.set_xlim(0, 1)
        ax.set_yticklabels("")
        ax.set_ylabel(variable_predict)
        ax.set_title(titulo)
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(['{:.0%}'.format(x) for x in ticks_loc])

        countv = 0
        for v in df_dist.iloc[:, 0]:
            ax.text(
                np.mean([countv, countv + v]) - 0.03, 1,
                '{:.1%}'.format(v), color='black', fontweight='bold',
            )
            countv = countv + v
        ax.legend(
            loc='upper center', bbox_to_anchor=(1.08, 1),
            shadow=True, ncol=1,
        )

    def plot_poder_predictivo_categorica(self, var, ax=None):
        """Distribución de una variable categórica según la clase objetivo.

        *** Gráfico adaptado de ModMC.py — poder_predictivo_categorica() ***

        Args:
            var: Nombre de la variable categórica a analizar.
            ax: Axes de matplotlib opcional. Si es None, crea una nueva figura.
        """
        plt.style.use('seaborn-v0_8-bright')
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(15, 10), dpi=200)
        data = self._df
        variable_predict = self.target
        df_cross = pd.crosstab(
            index=data[var], columns=data[variable_predict],
        )
        df_cross = df_cross.div(df_cross.sum(axis=1), axis=0)
        titulo = ("Distribución de la variable %s según la variable %s"
                  % (var, variable_predict))
        df_cross.plot(
            kind='barh', stacked=True, legend=True, ax=ax,
            xlim=(0, 1), title=titulo, width=0.8,
        )

        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(['{:.0%}'.format(x) for x in ticks_loc])
        ax.legend(
            loc='upper center', bbox_to_anchor=(1.08, 1),
            shadow=True, ncol=1,
        )

        for bars in ax.containers:
            plt.setp(bars, width=.9)
        for i in range(df_cross.shape[0]):
            countv = 0
            for v in df_cross.iloc[i]:
                ax.text(
                    np.mean([countv, countv + v]) - 0.03, i,
                    '{:.1%}'.format(v), color='black', fontweight='bold',
                )
                countv = countv + v

    def plot_poder_predictivo_numerica(self, var):
        """Distribución de una variable numérica por clase usando KDE.

        *** Gráfico adaptado de ModMC.py — poder_predictivo_numerica() ***

        Args:
            var: Nombre de la variable numérica a analizar.
        """
        plt.style.use('seaborn-v0_8-bright')
        sns.FacetGrid(
            self._df, hue=self.target, height=8, aspect=1.8,
        ).map(sns.kdeplot, var, fill=True).add_legend()

    # ═══════════════════════════════════════════════════════════════════════
    # ███  GRÁFICOS PROPIOS DE KNN                                        ███
    # ═══════════════════════════════════════════════════════════════════════

    def plot_matriz_confusion(self, titulo='Matriz de Confusión — KNN'):
        """Genera un heatmap de la matriz de confusión.

        Args:
            titulo: Título del gráfico. Por defecto 'Matriz de Confusión — KNN'.
        """
        plt.style.use('seaborn-v0_8-bright')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            self.__matriz_confusion, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.__nombres_clases,
            yticklabels=self.__nombres_clases, ax=ax,
        )
        ax.set_title(titulo)
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Valor Real')

    def plot_accuracy_vs_k(self, k_range=range(1, 31),
                           titulo='Exactitud vs. k — KNN'):
        """Gráfico de exactitud en función de distintos valores de k.

        Útil para identificar visualmente el k óptimo antes de usar
        buscar_mejor_k().

        Args:
            k_range: Rango de valores de k a evaluar. Por defecto range(1, 31).
            titulo: Título del gráfico. Por defecto 'Exactitud vs. k — KNN'.
        """
        plt.style.use('seaborn-v0_8-bright')
        if self.__X_train_escalado is None:
            self.preparar_datos()

        accuracies = []
        for k in k_range:
            knn = KNeighborsClassifier(
                n_neighbors=k, weights=self.__weights,
                metric=self.__metric,
            )
            knn.fit(self.__X_train_escalado, self.__y_train)
            accuracies.append(
                knn.score(self.__X_test_escalado, self.__y_test)
            )

        plt.figure(figsize=(10, 6))
        plt.plot(list(k_range), accuracies, marker='o', color='steelblue')
        plt.axvline(
            x=self.__n_neighbors, color='tomato', linestyle='--',
            label=f'k actual = {self.__n_neighbors}',
        )
        plt.title(titulo)
        plt.xlabel('Número de vecinos (k)')
        plt.ylabel('Exactitud')
        plt.legend()

    def __str__(self):
        if self.__accuracy is not None:
            return (
                f'KNN(n_neighbors={self.__n_neighbors}, '
                f'weights={self.__weights!r}, '
                f'metric={self.__metric!r}, '
                f'accuracy={self.__accuracy:.4f})'
            )
        return (
            f'KNN(n_neighbors={self.__n_neighbors}, '
            f'weights={self.__weights!r}, '
            f'metric={self.__metric!r}, '
            f'no ajustado)'
        )
