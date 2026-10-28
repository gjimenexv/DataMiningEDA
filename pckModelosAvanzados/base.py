"""Clase base de los modelos supervisados de Minería de Datos Avanzada.

Concentra todo lo que es común a KNN, árboles, ensambles y análisis
discriminante: preparación de datos, entrenamiento, métricas, validación
cruzada, búsqueda de hiperparámetros, umbral económico y gráficos.
Cada modelo concreto solo debe implementar ``_crear_modelo``.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

_ESCALADORES = {"standard": StandardScaler, "minmax": MinMaxScaler}


class ModeloBase:
    """Abstracción común de un clasificador supervisado de scikit-learn.

    El flujo de uso es siempre el mismo para todos los modelos::

        m = Modelo(df, target="y", ...)
        m.preparar_datos()
        m.ajustar()
        m.evaluar()

    El escalado (si aplica) se guarda dentro de un ``Pipeline``, de modo que
    se ajusta solo con los datos de entrenamiento y nunca hay fuga de
    información hacia el conjunto de prueba ni hacia la validación cruzada.
    """

    nombre = "Modelo"

    def __init__(self, df, target, scaler=None, random_state=42, **hiperparametros):
        """Inicializa el modelo con un DataFrame y la variable objetivo.

        Args:
            df: DataFrame de pandas con las variables predictoras y el target.
                Las predictoras deben ser numéricas.
            target: Nombre de la columna objetivo (clase a predecir).
            scaler: ``None`` (sin escalar), ``'standard'`` (z-score) o
                ``'minmax'`` (rango [0, 1]).
            random_state: Semilla para reproducibilidad. Por defecto 42.
            **hiperparametros: Hiperparámetros del estimador de scikit-learn
                (por ejemplo ``max_depth=5``). Se pasan tal cual al crear el modelo.

        Raises:
            ValueError: Si ``target`` no está en el DataFrame o el escalador
                no es válido.
        """
        if target not in df.columns:
            raise ValueError(f"La columna objetivo '{target}' no existe en el DataFrame.")
        if scaler is not None and scaler not in _ESCALADORES:
            raise ValueError("scaler debe ser None, 'standard' o 'minmax'.")
        self._df = df
        self._target = target
        self._scaler = scaler
        self._random_state = random_state
        self._hp = dict(hiperparametros)
        self._X = self._y = None
        self._X_train = self._X_test = self._y_train = self._y_test = None
        self._modelo = None
        self._predicciones = None
        self._probabilidades = None
        self._mejores_params = None
        self._busqueda = None
        self._umbral = 0.5

    # ─── Propiedades de solo lectura ────────────────────────────────────────

    @property
    def df(self):
        """DataFrame original con el que se creó el modelo."""
        return self._df

    @property
    def target(self):
        """Nombre de la columna objetivo."""
        return self._target

    @property
    def modelo(self):
        """Pipeline de scikit-learn ajustado (escalador + estimador)."""
        return self._modelo

    @property
    def estimador(self):
        """Estimador final de scikit-learn, sin el escalador."""
        return None if self._modelo is None else self._modelo[-1]

    @property
    def X_train(self):
        """Variables predictoras de entrenamiento (sin escalar)."""
        return self._X_train

    @property
    def X_test(self):
        """Variables predictoras de prueba (sin escalar)."""
        return self._X_test

    @property
    def y_train(self):
        """Etiquetas de entrenamiento."""
        return self._y_train

    @property
    def y_test(self):
        """Etiquetas de prueba."""
        return self._y_test

    @property
    def predicciones(self):
        """Predicciones sobre el conjunto de prueba (usan el umbral actual)."""
        return self._predicciones

    @property
    def probabilidades(self):
        """Probabilidades por clase sobre el conjunto de prueba."""
        return self._probabilidades

    @property
    def clases(self):
        """Lista ordenada de las clases de la variable objetivo."""
        return sorted(self._df[self._target].unique().tolist())

    @property
    def mejores_params(self):
        """Mejores hiperparámetros hallados por ``buscar_hiperparametros``."""
        return self._mejores_params

    @property
    def umbral(self):
        """Umbral de decisión (solo binario). 0.5 por defecto."""
        return self._umbral

    @property
    def matriz_confusion(self):
        """Matriz de confusión sobre el conjunto de prueba."""
        self._verificar_ajustado()
        return confusion_matrix(self._y_test, self._predicciones, labels=self.clases)

    @property
    def reporte(self):
        """Reporte de clasificación (precision, recall, F1 por clase) como texto."""
        self._verificar_ajustado()
        return classification_report(self._y_test, self._predicciones, zero_division=0)

    # ─── Construcción del modelo ────────────────────────────────────────────

    def _crear_modelo(self):
        """Crea el estimador de scikit-learn con los hiperparámetros actuales.

        Debe ser implementado por cada modelo concreto.

        Returns:
            Un estimador de scikit-learn sin ajustar.
        """
        raise NotImplementedError

    def _crear_pipeline(self):
        """Envuelve el estimador en un Pipeline con el escalador configurado.

        Returns:
            ``Pipeline`` de scikit-learn sin ajustar.
        """
        pasos = []
        if self._scaler is not None:
            pasos.append(("escalador", _ESCALADORES[self._scaler]()))
        pasos.append(("modelo", self._crear_modelo()))
        return Pipeline(pasos)

    def _verificar_datos(self):
        """Prepara los datos con valores por defecto si aún no se hizo."""
        if self._X_train is None:
            self.preparar_datos()

    def _verificar_ajustado(self):
        """Lanza un error claro si el modelo todavía no fue entrenado."""
        if self._modelo is None:
            raise RuntimeError("El modelo no está ajustado. Llame primero a ajustar().")

    # ─── Flujo principal ────────────────────────────────────────────────────

    def preparar_datos(self, test_size=0.2, estratificar=True):
        """Separa predictoras y target y divide en entrenamiento y prueba.

        Args:
            test_size: Proporción del conjunto de prueba. Por defecto 0.2
                (split 80/20 usado en las tareas del curso).
            estratificar: Si es True mantiene la proporción de clases en
                ambos conjuntos (recomendado, sobre todo con desbalance).
        """
        self._X = self._df.drop(columns=[self._target])
        self._y = self._df[self._target]
        (self._X_train, self._X_test,
         self._y_train, self._y_test) = train_test_split(
            self._X, self._y, test_size=test_size,
            random_state=self._random_state,
            stratify=self._y if estratificar else None,
        )
        self._modelo = None

    def ajustar(self):
        """Entrena el modelo con el conjunto de entrenamiento.

        Calcula además las predicciones y probabilidades sobre el conjunto
        de prueba y restablece el umbral a 0.5.
        """
        self._verificar_datos()
        self._modelo = self._crear_pipeline()
        self._modelo.fit(self._X_train, self._y_train)
        self._umbral = 0.5
        self._calcular_predicciones()

    def _calcular_predicciones(self):
        """Actualiza ``predicciones`` y ``probabilidades`` sobre el test."""
        if hasattr(self._modelo, "predict_proba"):
            self._probabilidades = self._modelo.predict_proba(self._X_test)
        self._predicciones = self._predecir_con_umbral(self._X_test)

    def _predecir_con_umbral(self, X):
        """Predice clases; en binario aplica el umbral sobre la clase positiva."""
        if self._es_binario() and self._umbral != 0.5:
            p = self._modelo.predict_proba(X)[:, 1]
            return np.where(p >= self._umbral, self.clases[1], self.clases[0])
        return self._modelo.predict(X)

    def _es_binario(self):
        """Indica si el problema tiene exactamente dos clases."""
        return len(self.clases) == 2

    def predecir(self, X_nuevo):
        """Predice la clase de observaciones nuevas.

        Args:
            X_nuevo: DataFrame con las mismas columnas predictoras que se
                usaron para entrenar.

        Returns:
            ``numpy.ndarray`` con la clase predicha de cada fila.
        """
        self._verificar_ajustado()
        return self._predecir_con_umbral(X_nuevo)

    def predecir_proba(self, X_nuevo):
        """Probabilidad de pertenecer a cada clase para observaciones nuevas.

        Args:
            X_nuevo: DataFrame con las columnas predictoras.

        Returns:
            Matriz (n_filas, n_clases) con las probabilidades.
        """
        self._verificar_ajustado()
        return self._modelo.predict_proba(X_nuevo)

    # ─── Evaluación ─────────────────────────────────────────────────────────

    def evaluar(self):
        """Calcula las métricas de la Etapa 4 del Credit Default Challenge.

        Returns:
            dict con ``accuracy``, ``balanced_accuracy``, ``auc`` (solo
            binario y si hay probabilidades) y, por cada clase,
            ``precision_<clase>``, ``recall_<clase>`` y ``f1_<clase>``.
        """
        self._verificar_ajustado()
        y, p = self._y_test, self._predicciones
        prec, rec, f1, _ = precision_recall_fscore_support(
            y, p, labels=self.clases, zero_division=0)
        res = {
            "accuracy": accuracy_score(y, p),
            "balanced_accuracy": balanced_accuracy_score(y, p),
        }
        for c, a, b, d in zip(self.clases, prec, rec, f1):
            res[f"precision_{c}"], res[f"recall_{c}"], res[f"f1_{c}"] = a, b, d
        if self._es_binario() and self._probabilidades is not None:
            res["auc"] = roc_auc_score(y, self._probabilidades[:, 1])
        return res

    def indices_binarios(self):
        """Desglosa la matriz de confusión 2x2 (VP, VN, FP, FN) y tasas.

        Returns:
            dict con VP, VN, FP, FN, sensibilidad (recall), especificidad,
            precision y F1; ``None`` si el problema no es binario.
        """
        self._verificar_ajustado()
        if not self._es_binario():
            return None
        vn, fp, fn, vp = self.matriz_confusion.ravel()
        div = lambda a, b: a / b if b else 0.0
        precision, recall = div(vp, vp + fp), div(vp, vp + fn)
        return {
            "VP": vp, "VN": vn, "FP": fp, "FN": fn,
            "sensibilidad": recall,
            "especificidad": div(vn, vn + fp),
            "precision": precision,
            "f1": div(2 * precision * recall, precision + recall),
        }

    def validacion_cruzada(self, cv=5, scoring=("accuracy", "f1_macro")):
        """Validación cruzada estratificada sobre el conjunto de entrenamiento.

        Args:
            cv: Número de folds. Por defecto 5.
            scoring: Métricas de scikit-learn a calcular.

        Returns:
            DataFrame con la media y la desviación estándar de cada métrica,
            como se pide en las tareas (media ± desviación).
        """
        self._verificar_datos()
        folds = StratifiedKFold(cv, shuffle=True, random_state=self._random_state)
        res = cross_validate(self._crear_pipeline(), self._X_train, self._y_train,
                             cv=folds, scoring=list(scoring), n_jobs=-1)
        filas = {m: {"media": res[f"test_{m}"].mean(), "desv_std": res[f"test_{m}"].std()}
                 for m in scoring}
        return pd.DataFrame(filas).T

    # ─── Hiperparámetros ────────────────────────────────────────────────────

    def buscar_hiperparametros(self, param_grid, metodo="grid", cv=5,
                               scoring="f1_macro", n_iter=30):
        """Optimiza hiperparámetros con validación cruzada sobre el train.

        Los nombres en ``param_grid`` son los del estimador de scikit-learn
        (por ejemplo ``{'max_depth': [3, 5, None]}``); el prefijo del
        Pipeline se agrega automáticamente. Al terminar, el modelo queda
        reentrenado con la mejor combinación.

        Args:
            param_grid: dict nombre -> lista de valores (o distribución para
                ``'random'`` y ``'bayes'``).
            metodo: ``'grid'`` (todas las combinaciones), ``'random'``
                (muestreo aleatorio) o ``'bayes'`` (requiere ``scikit-optimize``).
            cv: Número de folds estratificados. Por defecto 5.
            scoring: Métrica a maximizar. Para el desafío de default use ``'f1'``.
            n_iter: Combinaciones a probar en ``'random'`` y ``'bayes'``.

        Returns:
            dict con la mejor combinación de hiperparámetros.

        Raises:
            ValueError: Si el método no es válido.
            ImportError: Si se pide ``'bayes'`` sin ``scikit-optimize``.
        """
        self._verificar_datos()
        grid = {f"modelo__{k}": v for k, v in param_grid.items()}
        folds = StratifiedKFold(cv, shuffle=True, random_state=self._random_state)
        base = self._crear_pipeline()
        if metodo == "grid":
            busq = GridSearchCV(base, grid, cv=folds, scoring=scoring, n_jobs=-1)
        elif metodo == "random":
            busq = RandomizedSearchCV(base, grid, n_iter=n_iter, cv=folds,
                                      scoring=scoring, n_jobs=-1,
                                      random_state=self._random_state)
        elif metodo == "bayes":
            try:
                from skopt import BayesSearchCV
            except ImportError as e:
                raise ImportError("metodo='bayes' requiere: pip install scikit-optimize") from e
            busq = BayesSearchCV(base, grid, n_iter=n_iter, cv=folds, scoring=scoring,
                                 n_jobs=-1, random_state=self._random_state)
        else:
            raise ValueError("metodo debe ser 'grid', 'random' o 'bayes'.")
        busq.fit(self._X_train, self._y_train)
        self._busqueda = busq
        self._mejores_params = {k.replace("modelo__", ""): v
                                for k, v in busq.best_params_.items()}
        self._aplicar_params(self._mejores_params)
        self.ajustar()
        return self._mejores_params

    def _aplicar_params(self, params):
        """Actualiza los hiperparámetros que usará ``_crear_modelo``."""
        self._hp.update(params)

    @property
    def hiperparametros(self):
        """Copia de los hiperparámetros configurados para el estimador."""
        return dict(self._hp)

    def configurar(self, **hiperparametros):
        """Cambia hiperparámetros; hay que volver a llamar a ``ajustar()``.

        Args:
            **hiperparametros: Nombres y valores del estimador de scikit-learn.
        """
        self._aplicar_params(hiperparametros)
        self._modelo = None

    def resultados_busqueda(self):
        """Tabla con todas las combinaciones evaluadas por la última búsqueda.

        Returns:
            DataFrame ordenado por el ranking de la búsqueda.
        """
        if self._busqueda is None:
            raise RuntimeError("Aún no se ha ejecutado buscar_hiperparametros().")
        return pd.DataFrame(self._busqueda.cv_results_).sort_values("rank_test_score")

    # ─── Umbral de decisión (solo binario) ──────────────────────────────────

    def optimizar_umbral(self, costo_fp=1.0, costo_fn=1.0, pasos=99):
        """Busca el umbral que minimiza el costo total de errores.

        Reproduce el análisis de la Tarea 1 y del caso Vértice Retail:
        mismo modelo, distinto umbral, distinto costo de negocio. El umbral
        encontrado queda aplicado a ``predicciones``.

        Args:
            costo_fp: Costo de un falso positivo.
            costo_fn: Costo de un falso negativo.
            pasos: Cantidad de umbrales entre 0.01 y 0.99 a simular.

        Returns:
            DataFrame con umbral, costo, recall y precision de cada simulación.

        Raises:
            ValueError: Si el problema no es binario.
        """
        self._verificar_ajustado()
        if not self._es_binario():
            raise ValueError("optimizar_umbral solo aplica a problemas binarios.")
        p = self._probabilidades[:, 1]
        y = (self._y_test == self.clases[1]).to_numpy()
        filas = []
        for u in np.linspace(0.01, 0.99, pasos):
            pred = p >= u
            fp, fn, vp = (pred & ~y).sum(), (~pred & y).sum(), (pred & y).sum()
            filas.append({
                "umbral": u,
                "costo": fp * costo_fp + fn * costo_fn,
                "recall": vp / max(vp + fn, 1),
                "precision": vp / max(vp + fp, 1),
            })
        tabla = pd.DataFrame(filas)
        self._umbral = float(tabla.loc[tabla["costo"].idxmin(), "umbral"])
        self._calcular_predicciones()
        return tabla

    def fijar_umbral(self, umbral):
        """Fija manualmente el umbral de decisión (solo binario).

        Args:
            umbral: Valor entre 0 y 1. Bajarlo sube el recall y baja la precision.
        """
        self._verificar_ajustado()
        if not self._es_binario():
            raise ValueError("fijar_umbral solo aplica a problemas binarios.")
        self._umbral = float(umbral)
        self._calcular_predicciones()

    # ─── Gráficos ───────────────────────────────────────────────────────────

    def plot_matriz_confusion(self, ax=None):
        """Dibuja la matriz de confusión del conjunto de prueba.

        Args:
            ax: Axes de matplotlib opcional.
        """
        self._verificar_ajustado()
        ConfusionMatrixDisplay(self.matriz_confusion, display_labels=self.clases
                               ).plot(ax=ax, cmap="Blues", colorbar=False)
        (ax or plt.gca()).set_title(f"Matriz de confusión — {self.nombre}")

    def plot_curva_roc(self, ax=None):
        """Dibuja la curva ROC (solo binario).

        Args:
            ax: Axes de matplotlib opcional.
        """
        self._verificar_ajustado()
        if not self._es_binario():
            raise ValueError("plot_curva_roc solo aplica a problemas binarios.")
        ax = ax or plt.gca()
        fpr, tpr, _ = roc_curve(self._y_test, self._probabilidades[:, 1])
        auc = roc_auc_score(self._y_test, self._probabilidades[:, 1])
        ax.plot(fpr, tpr, label=f"{self.nombre} (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_ylabel("Tasa de verdaderos positivos")
        ax.set_title("Curva ROC")
        ax.legend()

    def importancia_variables(self):
        """Importancia de cada variable según el estimador, si la ofrece.

        Returns:
            Series ordenada de mayor a menor importancia.

        Raises:
            AttributeError: Si el modelo no expone ``feature_importances_``.
        """
        self._verificar_ajustado()
        imp = getattr(self.estimador, "feature_importances_", None)
        if imp is None:
            raise AttributeError(f"{self.nombre} no calcula importancia de variables.")
        return pd.Series(imp, index=self._X_train.columns).sort_values(ascending=False)

    def plot_importancia_variables(self, top=15, ax=None):
        """Gráfico de barras con las variables más importantes.

        Args:
            top: Cantidad de variables a mostrar. Por defecto 15.
            ax: Axes de matplotlib opcional.
        """
        imp = self.importancia_variables().head(top).sort_values()
        ax = ax or plt.gca()
        imp.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title(f"Importancia de variables — {self.nombre}")

    def __str__(self):
        """Resumen corto: nombre, estado y exactitud en el test."""
        if self._modelo is None:
            return f"{self.nombre}(no ajustado)"
        return f"{self.nombre}(accuracy={accuracy_score(self._y_test, self._predicciones):.4f})"
