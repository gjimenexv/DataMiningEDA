from .base import Supervisado

import math
import statistics
import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    LassoCV,
    Ridge,
    RidgeCV,
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")


class Regresion(Supervisado):
    """Clase para regresión supervisada con múltiples algoritmos.

    Hereda de Supervisado y agrega métodos para entrenar y evaluar modelos
    de regresión: Lineal Simple, Lineal Múltiple, Lasso, Ridge, SVM,
    Árbol de Decisión, Random Forest y Gradient Boosting, así como
    comparaciones entre algoritmos.

    Adaptada a partir de GuiaRegresiones.py del profesor Juan Murillo Morera.
    """

    def __init__(self, df, target='target'):
        """Inicializa la clase con un DataFrame y la variable objetivo.

        Args:
            df: DataFrame de pandas con los datos a analizar.
            target: Nombre de la columna que contiene la variable a predecir.
                    Por defecto 'target'.
        """
        super().__init__(df, target)

    # ─── Métricas de error ─────────────────────────────────────────────────

    @staticmethod
    def errores(y_true, y_pred):
        """Calcula RMSE, MAE y Error Relativo entre valores reales y predichos.

        *** Método adaptado de GuiaRegresiones.py — Prof. Juan Murillo Morera ***

        Args:
            y_true: Valores reales (array-like).
            y_pred: Valores predichos (array-like).

        Returns:
            pd.DataFrame: DataFrame con columnas 'Tipo' y 'Error' que contiene
                          RMSE (raíz del error cuadrático medio),
                          MAE  (error absoluto medio) y
                          ER   (error relativo).
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        n = len(y_true)
        RMSE = math.sqrt(np.sum(np.square(y_true - y_pred)) / n)
        MAE = np.sum(np.abs(y_true - y_pred)) / n
        ER = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
        return pd.DataFrame({
            'Tipo': ['RMSE', 'MAE', 'ER'],
            'Error': [RMSE, MAE, ER],
        })

    def evaluar(self, y_test, y_pred):
        """Imprime y devuelve las métricas de error para regresión.

        Sobreescribe el método de Supervisado con métricas apropiadas
        para regresión (RMSE, MAE, ER) en lugar de matriz de confusión.

        Args:
            y_test: Valores reales del conjunto de prueba.
            y_pred: Predicciones del modelo.

        Returns:
            pd.DataFrame: DataFrame con las métricas de error.
        """
        err = self.errores(y_test, y_pred)
        print(err.to_string(index=False))
        return err

    # ─── Estadísticas descriptivas ─────────────────────────────────────────

    def resumen_var_pred(self):
        """Imprime estadísticas descriptivas de la variable objetivo.

        *** Método adaptado de GuiaRegresiones.py — Prof. Juan Murillo Morera ***
        """
        y = self.df[self.target]
        cuartiles = statistics.quantiles(y)
        print(f"Máximo:    {np.max(y):.4f}")
        print(f"Q3:        {cuartiles[2]:.4f}")
        print(f"Mediana:   {cuartiles[1]:.4f}")
        print(f"Q1:        {cuartiles[0]:.4f}")
        print(f"Mínimo:    {np.min(y):.4f}")

    # ─── Algoritmos de regresión ───────────────────────────────────────────

    def regresion_lineal_simple(self, feature):
        """Regresión lineal simple con una sola variable predictora.

        *** Método adaptado de GuiaRegresiones.py — Prof. Juan Murillo Morera ***

        Args:
            feature: Nombre de la columna a usar como único predictor.

        Returns:
            pd.DataFrame: DataFrame con las métricas de error.
        """
        if self.X_train_escalado is None:
            self.preparar_datos()
        X_tr = self.X_train_escalado[[feature]].values
        X_te = self.X_test_escalado[[feature]].values
        modelo = LinearRegression().fit(X_tr, self.y_train)
        print(f"Coeficiente w1: {modelo.coef_}")
        print(f"Coeficiente w0: {modelo.intercept_:.4f}")
        predicciones = modelo.predict(X_te)
        print(f"Primeras 3 predicciones: {predicciones[:3]}")
        return self.errores(self.y_test, predicciones)

    def regresion_lineal_multiple(self):
        """Regresión lineal múltiple con todas las variables predictoras.

        *** Método adaptado de GuiaRegresiones.py — Prof. Juan Murillo Morera ***

        Returns:
            pd.DataFrame: DataFrame con las métricas de error.
        """
        if self.X_train_escalado is None:
            self.preparar_datos()
        modelo = LinearRegression()
        modelo.fit(self.X_train_escalado, self.y_train)
        predicciones = modelo.predict(self.X_test_escalado)
        print(f"Primeras 3 predicciones: {predicciones[:3]}")
        return self.errores(self.y_test, predicciones)

    def regresion_lasso(self, alpha=0.1):
        """Regresión Lasso (L1) con alpha fijo.

        *** Método adaptado de GuiaRegresiones.py — Prof. Juan Murillo Morera ***

        Args:
            alpha: Parámetro de regularización. Por defecto 0.1.

        Returns:
            pd.DataFrame: DataFrame con las métricas de error.
        """
        if self.X_train_escalado is None:
            self.preparar_datos()
        modelo = Lasso(alpha=alpha)
        modelo.fit(self.X_train_escalado, self.y_train)
        predicciones = modelo.predict(self.X_test_escalado)
        df_coef = pd.DataFrame({
            'predictor': self.X_train_escalado.columns,
            'coef': modelo.coef_.flatten(),
        })
        print(df_coef.to_string(index=False))
        return self.errores(self.y_test, predicciones)

    def regresion_lasso_cv(self, cv=10):
        """Regresión Lasso con alpha óptimo seleccionado por validación cruzada.

        *** Método adaptado de GuiaRegresiones.py — Prof. Juan Murillo Morera ***

        Args:
            cv: Número de folds para validación cruzada. Por defecto 10.

        Returns:
            pd.DataFrame: DataFrame con las métricas de error.
        """
        if self.X_train_escalado is None:
            self.preparar_datos()
        modelo_cv = LassoCV(alphas=np.logspace(-6, 6, 200), cv=cv)
        modelo_cv.fit(self.X_train_escalado, self.y_train)
        print(f"Alpha óptimo: {modelo_cv.alpha_:.6f}")
        modelo = Lasso(alpha=modelo_cv.alpha_)
        modelo.fit(self.X_train_escalado, self.y_train)
        predicciones = modelo.predict(self.X_test_escalado)
        df_coef = pd.DataFrame({
            'predictor': self.X_train_escalado.columns,
            'coef': modelo.coef_.flatten(),
        })
        print(df_coef.to_string(index=False))
        return self.errores(self.y_test, predicciones)

    def regresion_ridge(self, alpha=1.0):
        """Regresión Ridge (L2) con alpha fijo.

        *** Método adaptado de GuiaRegresiones.py — Prof. Juan Murillo Morera ***

        Args:
            alpha: Parámetro de regularización. Por defecto 1.0.

        Returns:
            pd.DataFrame: DataFrame con las métricas de error.
        """
        if self.X_train_escalado is None:
            self.preparar_datos()
        modelo = Ridge(alpha=alpha)
        modelo.fit(self.X_train_escalado, self.y_train)
        predicciones = modelo.predict(self.X_test_escalado)
        df_coef = pd.DataFrame({
            'predictor': self.X_train_escalado.columns,
            'coef': modelo.coef_.flatten(),
        })
        print(df_coef.to_string(index=False))
        return self.errores(self.y_test, predicciones)

    def regresion_ridge_cv(self):
        """Regresión Ridge con alpha óptimo seleccionado por validación cruzada.

        *** Método adaptado de GuiaRegresiones.py — Prof. Juan Murillo Morera ***

        Returns:
            pd.DataFrame: DataFrame con las métricas de error.
        """
        if self.X_train_escalado is None:
            self.preparar_datos()
        modelo_cv = RidgeCV(
            alphas=np.logspace(-10, 2, 200),
            fit_intercept=True,
            store_cv_results=True,
        )
        modelo_cv.fit(self.X_train_escalado, self.y_train)
        print(f"Alpha óptimo: {modelo_cv.alpha_:.6f}")
        modelo = Ridge(alpha=modelo_cv.alpha_)
        modelo.fit(self.X_train_escalado, self.y_train)
        predicciones = modelo.predict(self.X_test_escalado)
        df_coef = pd.DataFrame({
            'predictor': self.X_train_escalado.columns,
            'coef': modelo.coef_.flatten(),
        })
        print(df_coef.to_string(index=False))
        return self.errores(self.y_test, predicciones)

    def regresion_svm(self, kernel='rbf', C=100, epsilon=0.1, degree=3):
        """Regresión con Support Vector Machine (SVR).

        *** Método adaptado de GuiaRegresiones.py — Prof. Juan Murillo Morera ***

        Args:
            kernel: Kernel del SVR ('rbf', 'linear', 'poly'). Por defecto 'rbf'.
            C: Parámetro de regularización. Por defecto 100.
            epsilon: Margen de tolerancia del tubo. Por defecto 0.1.
            degree: Grado del polinomio (solo kernel 'poly'). Por defecto 3.

        Returns:
            pd.DataFrame: DataFrame con las métricas de error.
        """
        if self.X_train_escalado is None:
            self.preparar_datos()
        modelo = SVR(kernel=kernel, C=C, epsilon=epsilon, degree=degree)
        modelo.fit(self.X_train_escalado, self.y_train)
        predicciones = modelo.predict(self.X_test_escalado)
        return self.errores(self.y_test, predicciones)

    def regresion_arbol(self, max_depth=3, random_state=123):
        """Regresión con Árbol de Decisión.

        *** Método adaptado de GuiaRegresiones.py — Prof. Juan Murillo Morera ***

        Args:
            max_depth: Profundidad máxima del árbol. Por defecto 3.
            random_state: Semilla para reproducibilidad. Por defecto 123.

        Returns:
            pd.DataFrame: DataFrame con las métricas de error.
        """
        if self.X_train_escalado is None:
            self.preparar_datos()
        modelo = DecisionTreeRegressor(
            max_depth=max_depth, random_state=random_state
        )
        modelo.fit(self.X_train_escalado, self.y_train)
        predicciones = modelo.predict(self.X_test_escalado)
        return self.errores(self.y_test, predicciones)

    def regresion_bosque(self, max_depth=2, random_state=0):
        """Regresión con Random Forest.

        *** Método adaptado de GuiaRegresiones.py — Prof. Juan Murillo Morera ***

        Args:
            max_depth: Profundidad máxima de los árboles. Por defecto 2.
            random_state: Semilla para reproducibilidad. Por defecto 0.

        Returns:
            pd.DataFrame: DataFrame con las métricas de error.
        """
        if self.X_train_escalado is None:
            self.preparar_datos()
        modelo = RandomForestRegressor(
            max_depth=max_depth, random_state=random_state
        )
        modelo.fit(self.X_train_escalado, self.y_train)
        predicciones = modelo.predict(self.X_test_escalado)
        return self.errores(self.y_test, predicciones)

    def regresion_xgboosting(self, n_estimators=500, max_depth=4,
                             min_samples_split=5):
        """Regresión con Gradient Boosting (XGBoosting).

        *** Método adaptado de GuiaRegresiones.py — Prof. Juan Murillo Morera ***

        Args:
            n_estimators: Número de estimadores. Por defecto 500.
            max_depth: Profundidad máxima. Por defecto 4.
            min_samples_split: Mínimo de muestras para dividir. Por defecto 5.

        Returns:
            pd.DataFrame: DataFrame con las métricas de error.
        """
        if self.X_train_escalado is None:
            self.preparar_datos()
        modelo = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        modelo.fit(self.X_train_escalado, self.y_train)
        predicciones = modelo.predict(self.X_test_escalado)
        return self.errores(self.y_test, predicciones)

    # ─── Comparaciones ─────────────────────────────────────────────────────

    def comparacion_lasso(self):
        """Compara Lasso con alpha fijo vs. Lasso con alpha óptimo (CV).

        *** Método adaptado de GuiaRegresiones.py — Prof. Juan Murillo Morera ***

        Returns:
            pd.DataFrame: Tabla comparativa con RMSE, MAE y ER.
        """
        print("─── Lasso (alpha=0.1) ───")
        err1 = self.regresion_lasso()
        print("\n─── Lasso Óptimo (CV) ───")
        err2 = self.regresion_lasso_cv()
        comparacion = pd.DataFrame(
            [err1.Error.values, err2.Error.values],
            columns=['RMSE', 'MAE', 'ER'],
            index=['Lasso', 'Lasso Óptimo'],
        )
        print(f"\n{comparacion}")
        return comparacion

    def comparacion_ridge(self):
        """Compara Ridge con alpha fijo vs. Ridge con alpha óptimo (CV).

        *** Método adaptado de GuiaRegresiones.py — Prof. Juan Murillo Morera ***

        Returns:
            pd.DataFrame: Tabla comparativa con RMSE, MAE y ER.
        """
        print("─── Ridge (alpha=1.0) ───")
        err1 = self.regresion_ridge()
        print("\n─── Ridge Óptimo (CV) ───")
        err2 = self.regresion_ridge_cv()
        comparacion = pd.DataFrame(
            [err1.Error.values, err2.Error.values],
            columns=['RMSE', 'MAE', 'ER'],
            index=['Ridge', 'Ridge Óptimo'],
        )
        print(f"\n{comparacion}")
        return comparacion

    def comparacion_svm(self):
        """Compara tres kernels de SVR: rbf, linear y poly.

        *** Método adaptado de GuiaRegresiones.py — Prof. Juan Murillo Morera ***

        Returns:
            pd.DataFrame: Tabla comparativa con RMSE, MAE y ER.
        """
        err1 = self.regresion_svm(kernel='rbf')
        err2 = self.regresion_svm(kernel='linear')
        err3 = self.regresion_svm(kernel='poly')
        comparacion = pd.DataFrame(
            [err1.Error.values, err2.Error.values, err3.Error.values],
            columns=['RMSE', 'MAE', 'ER'],
            index=['SVM rbf', 'SVM linear', 'SVM poly'],
        )
        print(comparacion)
        return comparacion

    def benchmark(self):
        """Compara todos los algoritmos de regresión e imprime una tabla resumen.

        Ejecuta: Lineal Múltiple, Lasso CV, Ridge CV, SVM (rbf, linear, poly),
        Árbol de Decisión, Random Forest y Gradient Boosting.

        *** Método adaptado de GuiaRegresiones.py — Prof. Juan Murillo Morera ***

        Returns:
            pd.DataFrame: Tabla comparativa con RMSE, MAE y ER para cada algoritmo.
        """
        if self.X_train_escalado is None:
            self.preparar_datos()

        resultados = []
        nombres = []

        algoritmos = [
            ("Múltiple",     self.regresion_lineal_multiple),
            ("Lasso CV",     self.regresion_lasso_cv),
            ("Ridge CV",     self.regresion_ridge_cv),
            ("Árbol",        self.regresion_arbol),
            ("Bosques",      self.regresion_bosque),
            ("Potenciación", self.regresion_xgboosting),
        ]

        for nombre, metodo in algoritmos:
            print(f"\n{'═' * 40}")
            print(f"  {nombre}")
            print(f"{'═' * 40}")
            err = metodo()
            resultados.append(err.Error.values)
            nombres.append(nombre)

        # SVM con tres kernels
        for kernel in ['rbf', 'linear', 'poly']:
            print(f"\n{'═' * 40}")
            print(f"  SVM {kernel}")
            print(f"{'═' * 40}")
            err = self.regresion_svm(kernel=kernel)
            resultados.append(err.Error.values)
            nombres.append(f"SVM {kernel}")

        comparacion = pd.DataFrame(
            resultados,
            columns=['RMSE', 'MAE', 'ER'],
            index=nombres,
        )
        print(f"\n{'═' * 60}")
        print("  RESUMEN COMPARATIVO")
        print(f"{'═' * 60}")
        print(comparacion)
        return comparacion
