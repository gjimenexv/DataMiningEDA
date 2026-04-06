from .base import Supervisado

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")


class Clasificacion(Supervisado):
    """Clase para clasificación supervisada con múltiples algoritmos.

    Hereda de Supervisado y agrega métodos para entrenar y evaluar
    modelos de clasificación: KNN, Decision Tree, Random Forest,
    Gradient Boosting y AdaBoost, así como un benchmark comparativo.

    Adaptada a partir de GuiaClaseSupervisada.py del profesor Juan Murillo Morera.
    """

    def __init__(self, df, target='target'):
        """Inicializa la clase con un DataFrame y la variable objetivo.

        Args:
            df: DataFrame de pandas con los datos a analizar.
            target: Nombre de la columna que contiene la variable clase.
        """
        super().__init__(df, target)

    # ─── Modelos privados ──────────────────────────────────────────────────

    def __modelo_knn(self, X_train, y_train, n_neighbors, algorithm):
        """Entrena un modelo KNN.

        Args:
            X_train: Features de entrenamiento.
            y_train: Etiquetas de entrenamiento.
            n_neighbors: Número de vecinos.
            algorithm: Algoritmo de búsqueda ('auto', 'ball_tree', 'kd_tree', 'brute').

        Returns:
            KNeighborsClassifier: Modelo entrenado.
        """
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors, algorithm=algorithm
        )
        model.fit(X_train, y_train)
        return model

    def __modelo_dt(self, X_train, y_train, min_samples_split, max_depth):
        """Entrena un modelo Decision Tree.

        Args:
            X_train: Features de entrenamiento.
            y_train: Etiquetas de entrenamiento.
            min_samples_split: Mínimo de muestras para dividir un nodo.
            max_depth: Profundidad máxima del árbol.

        Returns:
            DecisionTreeClassifier: Modelo entrenado.
        """
        model = DecisionTreeClassifier(
            min_samples_split=min_samples_split, max_depth=max_depth
        )
        model.fit(X_train, y_train)
        return model

    def __modelo_rf(self, X_train, y_train, n_estimators,
                    min_samples_split, max_depth):
        """Entrena un modelo Random Forest.

        Args:
            X_train: Features de entrenamiento.
            y_train: Etiquetas de entrenamiento.
            n_estimators: Número de árboles en el bosque.
            min_samples_split: Mínimo de muestras para dividir un nodo.
            max_depth: Profundidad máxima de los árboles.

        Returns:
            RandomForestClassifier: Modelo entrenado.
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            max_depth=max_depth,
        )
        model.fit(X_train, y_train)
        return model

    def __modelo_xg(self, X_train, y_train, n_estimators,
                    min_samples_split, max_depth):
        """Entrena un modelo Gradient Boosting.

        Args:
            X_train: Features de entrenamiento.
            y_train: Etiquetas de entrenamiento.
            n_estimators: Número de estimadores.
            min_samples_split: Mínimo de muestras para dividir un nodo.
            max_depth: Profundidad máxima de los árboles.

        Returns:
            GradientBoostingClassifier: Modelo entrenado.
        """
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            max_depth=max_depth,
        )
        model.fit(X_train, y_train)
        return model

    def __modelo_ada(self, X_train, y_train, estimator, n_estimators):
        """Entrena un modelo AdaBoost.

        Args:
            X_train: Features de entrenamiento.
            y_train: Etiquetas de entrenamiento.
            estimator: Estimador base (e.g. DecisionTreeClassifier).
            n_estimators: Número de estimadores.

        Returns:
            AdaBoostClassifier: Modelo entrenado.
        """
        model = AdaBoostClassifier(
            estimator=estimator, n_estimators=n_estimators
        )
        model.fit(X_train, y_train)
        return model

    def __predecir(self, model, X_test):
        """Genera predicciones con un modelo entrenado.

        Args:
            model: Modelo de scikit-learn entrenado.
            X_test: Features de prueba.

        Returns:
            numpy.ndarray: Predicciones.
        """
        return model.predict(X_test)

    # ─── Métodos públicos de clasificación ─────────────────────────────────

    def clasificar_knn(self, n_neighbors=5):
        """Entrena y evalúa un modelo KNN con los cuatro algoritmos de búsqueda.

        *** Método adaptado de GuiaClaseSupervisada.py — Prof. Juan Murillo Morera ***

        Args:
            n_neighbors: Número de vecinos. Por defecto 5.
        """
        algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
        for algorithm in algorithms:
            print(f"\nUsando algoritmo: {algorithm}")
            self.preparar_datos()
            model = self.__modelo_knn(
                self.X_train_escalado, self.y_train,
                n_neighbors, algorithm,
            )
            y_pred = self.__predecir(model, self.X_test_escalado)
            self.evaluar(self.y_test, y_pred)

    def clasificar_dt(self, min_samples_split=2, max_depth=4):
        """Entrena y evalúa un modelo Decision Tree.

        *** Método adaptado de GuiaClaseSupervisada.py — Prof. Juan Murillo Morera ***

        Args:
            min_samples_split: Mínimo de muestras para dividir. Por defecto 2.
            max_depth: Profundidad máxima. Por defecto 4.
        """
        self.preparar_datos()
        model = self.__modelo_dt(
            self.X_train_escalado, self.y_train,
            min_samples_split, max_depth,
        )
        y_pred = self.__predecir(model, self.X_test_escalado)
        self.evaluar(self.y_test, y_pred)

    def clasificar_rf(self, n_estimators=100, min_samples_split=2, max_depth=4):
        """Entrena y evalúa un modelo Random Forest.

        *** Método adaptado de GuiaClaseSupervisada.py — Prof. Juan Murillo Morera ***

        Args:
            n_estimators: Número de árboles. Por defecto 100.
            min_samples_split: Mínimo de muestras para dividir. Por defecto 2.
            max_depth: Profundidad máxima. Por defecto 4.
        """
        self.preparar_datos()
        model = self.__modelo_rf(
            self.X_train_escalado, self.y_train,
            n_estimators, min_samples_split, max_depth,
        )
        y_pred = self.__predecir(model, self.X_test_escalado)
        self.evaluar(self.y_test, y_pred)

    def clasificar_xg(self, n_estimators=100, min_samples_split=2, max_depth=4):
        """Entrena y evalúa un modelo Gradient Boosting.

        *** Método adaptado de GuiaClaseSupervisada.py — Prof. Juan Murillo Morera ***

        Args:
            n_estimators: Número de estimadores. Por defecto 100.
            min_samples_split: Mínimo de muestras para dividir. Por defecto 2.
            max_depth: Profundidad máxima. Por defecto 4.
        """
        self.preparar_datos()
        model = self.__modelo_xg(
            self.X_train_escalado, self.y_train,
            n_estimators, min_samples_split, max_depth,
        )
        y_pred = self.__predecir(model, self.X_test_escalado)
        self.evaluar(self.y_test, y_pred)

    def clasificar_ada(self, n_estimators=100):
        """Entrena y evalúa un modelo AdaBoost con múltiples estimadores base.

        Prueba automáticamente con Decision Tree, Random Forest y Gradient
        Boosting como estimadores base.

        *** Método adaptado de GuiaClaseSupervisada.py — Prof. Juan Murillo Morera ***

        Args:
            n_estimators: Número de estimadores. Por defecto 100.
        """
        estimators = {
            "Decision Tree": DecisionTreeClassifier(
                min_samples_split=2, max_depth=4
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, min_samples_split=2, max_depth=4
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, min_samples_split=2, max_depth=4
            ),
        }
        for name, estimator in estimators.items():
            print(f"\nUsando metodo: {name}")
            self.preparar_datos()
            model = self.__modelo_ada(
                self.X_train_escalado, self.y_train,
                estimator, n_estimators,
            )
            y_pred = self.__predecir(model, self.X_test_escalado)
            self.evaluar(self.y_test, y_pred)

    def ada_grid_search(self, param_grid):
        """Busca los mejores hiperparámetros para AdaBoost usando GridSearchCV.

        *** Método adaptado de GuiaClaseSupervisada.py — Prof. Juan Murillo Morera ***

        Args:
            param_grid: Diccionario con los hiperparámetros a buscar.
                        Ejemplo: {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
        """
        self.preparar_datos()
        ada = AdaBoostClassifier(estimator=DecisionTreeClassifier())
        grid_search = GridSearchCV(
            ada, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1
        )
        grid_search.fit(self.X_train_escalado, self.y_train)

        print("Mejores hiperparámetros encontrados:")
        print(grid_search.best_params_)
        print("\nMejor puntuación de validación cruzada:")
        print(grid_search.best_score_)
        print("\nRendimiento en el conjunto de prueba:")
        y_pred = grid_search.predict(self.X_test_escalado)
        self.evaluar(self.y_test, y_pred)

    # ─── Benchmark privados ────────────────────────────────────────────────

    def __knn_bm(self, n_neighbors=5, algorithm='auto'):
        """Benchmark interno para KNN."""
        self.preparar_datos()
        model = self.__modelo_knn(
            self.X_train_escalado, self.y_train, n_neighbors, algorithm
        )
        y_pred = self.__predecir(model, self.X_test_escalado)
        MC = confusion_matrix(self.y_test, y_pred)
        return self.indices_general(MC, list(np.unique(self.y)))

    def __dt_bm(self, min_samples_split=2, max_depth=4):
        """Benchmark interno para Decision Tree."""
        self.preparar_datos()
        model = self.__modelo_dt(
            self.X_train_escalado, self.y_train,
            min_samples_split, max_depth,
        )
        y_pred = self.__predecir(model, self.X_test_escalado)
        MC = confusion_matrix(self.y_test, y_pred)
        return self.indices_general(MC, list(np.unique(self.y)))

    def __rf_bm(self, n_estimators=100, min_samples_split=2, max_depth=4):
        """Benchmark interno para Random Forest."""
        self.preparar_datos()
        model = self.__modelo_rf(
            self.X_train_escalado, self.y_train,
            n_estimators, min_samples_split, max_depth,
        )
        y_pred = self.__predecir(model, self.X_test_escalado)
        MC = confusion_matrix(self.y_test, y_pred)
        return self.indices_general(MC, list(np.unique(self.y)))

    def __xg_bm(self, n_estimators=100, min_samples_split=2, max_depth=4):
        """Benchmark interno para Gradient Boosting."""
        self.preparar_datos()
        model = self.__modelo_xg(
            self.X_train_escalado, self.y_train,
            n_estimators, min_samples_split, max_depth,
        )
        y_pred = self.__predecir(model, self.X_test_escalado)
        MC = confusion_matrix(self.y_test, y_pred)
        return self.indices_general(MC, list(np.unique(self.y)))

    def __ada_bm(self, n_estimators=100):
        """Benchmark interno para AdaBoost."""
        self.preparar_datos()
        estimator = DecisionTreeClassifier(
            min_samples_split=2, max_depth=4
        )
        model = self.__modelo_ada(
            self.X_train_escalado, self.y_train,
            estimator, n_estimators,
        )
        y_pred = self.__predecir(model, self.X_test_escalado)
        MC = confusion_matrix(self.y_test, y_pred)
        return self.indices_general(MC, list(np.unique(self.y)))

    # ─── Benchmark comparativo ─────────────────────────────────────────────

    def benchmark(self):
        """Compara los cinco algoritmos de clasificación e imprime una tabla resumen.

        La tabla muestra precisión global (PG), error global (EG), precisión
        positiva (PP) y precisión negativa (PN) para cada algoritmo.

        Nota: PP y PN asumen clasificación binaria (dos clases).

        *** Método adaptado de GuiaClaseSupervisada.py — Prof. Juan Murillo Morera ***
        """
        datos = {
            "PG": [0, 0, 0, 0, 0],
            "EG": [0, 0, 0, 0, 0],
            "PP": [0, 0, 0, 0, 0],
            "PN": [0, 0, 0, 0, 0],
        }
        Tdatos = pd.DataFrame(
            datos,
            index=[
                "AlgKnn", "AlgDT", "AlgRF", "AlgXGBoost", "AlgADABoost",
            ],
            columns=["PG", "EG", "PP", "PN"],
        )

        algoritmos = [
            ("AlgKnn", self.__knn_bm),
            ("AlgDT", self.__dt_bm),
            ("AlgRF", self.__rf_bm),
            ("AlgXGBoost", self.__xg_bm),
            ("AlgADABoost", self.__ada_bm),
        ]

        for alg_name, alg_method in algoritmos:
            indices = alg_method()
            PP = indices['Precisión por categoría']
            Tdatos.loc[alg_name, "PG"] = indices['Precisión Global']
            Tdatos.loc[alg_name, "EG"] = indices['Error Global']
            Tdatos.loc[alg_name, "PP"] = (
                PP.iloc[0, 1] if PP.shape[1] > 1 else PP.iloc[0, 0]
            )
            Tdatos.loc[alg_name, "PN"] = PP.iloc[0, 0]

        print(Tdatos)
