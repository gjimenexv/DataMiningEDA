import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE as TSNE_sklearn
from sklearn.preprocessing import StandardScaler

from pckEDA.eda import AnalisisDatosExploratorio
from .base import NoSupervisado


class TSNE(NoSupervisado):
    """Reducción de dimensionalidad con t-SNE (t-distributed Stochastic Neighbor Embedding).

    Hereda de NoSupervisado → AnalisisDatosExploratorio, por lo que dispone de todos los
    métodos de carga, limpieza y transformación del EDA antes de ajustar el modelo.

    t-SNE proyecta datos de alta dimensionalidad a 2D o 3D preservando la estructura
    local (vecindades), lo que lo hace especialmente útil para visualización exploratoria
    y para revelar agrupaciones naturales en los datos.

    Nota: t-SNE no soporta proyección de nuevos datos tras el ajuste (no tiene transform()).
    Para proyectar datos nuevos, considera UMAP.

    Flujo de uso:
        1. Crear la instancia con la ruta del CSV.
        2. Aplicar los métodos heredados del EDA (codificarCategorica, eliminarNulos, etc.).
        3. Llamar a ajustar() para calcular la proyección sobre los datos limpios.
        4. Usar los métodos de visualización para explorar la proyección.
    """

    def __init__(self, path, num, n_componentes=2, perplejidad=30,
                 iteraciones=1000, tasa_aprendizaje='auto', random_state=42):
        """Carga los datos desde un CSV y configura los parámetros del modelo t-SNE.

        El modelo no se ajusta en este paso — primero aplica los métodos heredados del EDA
        para limpiar y preparar los datos, luego llama a ajustar().

        Args:
            path: Ruta del archivo CSV a cargar.
            num: Modo de lectura del CSV (1: separador coma con índice, 2: separador punto y coma).
            n_componentes: Dimensiones de la proyección resultante (2 o 3). Por defecto 2.
            perplejidad: Número aproximado de vecinos cercanos considerados por cada punto.
                         Valores típicos entre 5 y 50. Datasets grandes admiten valores mayores.
                         Por defecto 30.
            iteraciones: Número máximo de iteraciones de optimización. Por defecto 1000.
            tasa_aprendizaje: Tasa de aprendizaje del algoritmo. 'auto' es recomendado
                              (equivale a max(200, n_samples / 12)). Por defecto 'auto'.
            random_state: Semilla para reproducibilidad. Por defecto 42.
        """
        AnalisisDatosExploratorio.__init__(self, path, num)
        self.__n_componentes = n_componentes
        self.__perplejidad = perplejidad
        self.__iteraciones = iteraciones
        self.__tasa_aprendizaje = tasa_aprendizaje
        self.__random_state = random_state
        self.__datos_escalados = None
        self.__modelo = None
        self.__coordenadas = None

    def ajustar(self):
        """Calcula la proyección t-SNE sobre los datos actuales (self.df).

        Debe llamarse después de aplicar los métodos de limpieza y transformación del EDA.
        Estandariza los datos automáticamente con StandardScaler antes de la proyección.
        """
        escalador = StandardScaler()
        self.__datos_escalados = escalador.fit_transform(self.df)

        self.__modelo = TSNE_sklearn(
            n_components=self.__n_componentes,
            perplexity=self.__perplejidad,
            max_iter=self.__iteraciones,
            learning_rate=self.__tasa_aprendizaje,
            random_state=self.__random_state,
        )
        self.__coordenadas = self.__modelo.fit_transform(self.__datos_escalados)

    # ─── Properties ────────────────────────────────────────────────────────────

    @property
    def datos_escalados(self):
        """Devuelve el array numpy con los datos estandarizados (media 0, desviación 1)."""
        return self.__datos_escalados

    @datos_escalados.setter
    def datos_escalados(self, datos_escalados):
        """Args:
            datos_escalados: Nuevo array de datos estandarizados.
        """
        self.__datos_escalados = datos_escalados

    @property
    def modelo(self):
        """Devuelve el modelo TSNE de scikit-learn ajustado."""
        return self.__modelo

    @modelo.setter
    def modelo(self, modelo):
        """Args:
            modelo: Nuevo modelo TSNE ajustado.
        """
        self.__modelo = modelo

    @property
    def coordenadas(self):
        """Devuelve el array numpy con las coordenadas proyectadas (n_samples × n_componentes)."""
        return self.__coordenadas

    @coordenadas.setter
    def coordenadas(self, coordenadas):
        """Args:
            coordenadas: Nuevo array de coordenadas proyectadas.
        """
        self.__coordenadas = coordenadas

    @property
    def n_componentes(self):
        """Devuelve el número de dimensiones de la proyección (2 o 3)."""
        return self.__n_componentes

    @n_componentes.setter
    def n_componentes(self, n_componentes):
        """Args:
            n_componentes: Nuevo número de componentes.
        """
        self.__n_componentes = n_componentes

    @property
    def perplejidad(self):
        """Devuelve el valor de perplejidad utilizado en el ajuste."""
        return self.__perplejidad

    @perplejidad.setter
    def perplejidad(self, perplejidad):
        """Args:
            perplejidad: Nuevo valor de perplejidad.
        """
        self.__perplejidad = perplejidad

    @property
    def iteraciones(self):
        """Devuelve el número de iteraciones de optimización."""
        return self.__iteraciones

    @iteraciones.setter
    def iteraciones(self, iteraciones):
        """Args:
            iteraciones: Nuevo número de iteraciones.
        """
        self.__iteraciones = iteraciones

    @property
    def random_state(self):
        """Devuelve la semilla de aleatoriedad utilizada."""
        return self.__random_state

    @random_state.setter
    def random_state(self, random_state):
        """Args:
            random_state: Nueva semilla de aleatoriedad.
        """
        self.__random_state = random_state

    # ─── Visualization Methods ──────────────────────────────────────────────────

    def plot_proyeccion(self, etiquetas=None, titulo='Proyección t-SNE'):
        """Genera un diagrama de dispersión de la proyección 2D t-SNE.

        Opcionalmente colorea los puntos con etiquetas externas (e.g. clusters de K-Means o HAC)
        para revelar si la estructura de clustering coincide con la topología del embedding.

        Args:
            etiquetas: Array de etiquetas para colorear los puntos (e.g. hac.etiquetas,
                       kmeans.etiquetas). Si es None, todos los puntos se muestran en el
                       mismo color. Por defecto None.
            titulo: Título del gráfico. Por defecto 'Proyección t-SNE'.
        """
        plt.style.use('seaborn-v0_8-bright')
        x = self.__coordenadas[:, 0]
        y = self.__coordenadas[:, 1]
        if etiquetas is not None:
            n_grupos = len(np.unique(etiquetas))
            scatter = plt.scatter(x, y, c=etiquetas,
                                  cmap=plt.cm.get_cmap('tab10', n_grupos),
                                  alpha=0.6, s=15)
            plt.colorbar(scatter, label='Cluster')
        else:
            plt.scatter(x, y, color='steelblue', alpha=0.6, s=15)
        plt.title(titulo)
        plt.xlabel('t-SNE dimensión 1')
        plt.ylabel('t-SNE dimensión 2')

    def plot_perplejidad(self, valores=(5, 15, 30, 50), titulo='Efecto de la Perplejidad — t-SNE'):
        """Compara la proyección t-SNE con distintos valores de perplejidad en una cuadrícula.

        Útil para elegir el valor de perplejidad más adecuado antes del análisis final.

        Args:
            valores: Tupla o lista de valores de perplejidad a comparar. Por defecto (5, 15, 30, 50).
            titulo: Título general de la figura. Por defecto 'Efecto de la Perplejidad — t-SNE'.
        """
        plt.style.use('seaborn-v0_8-bright')
        n = len(valores)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        fig.suptitle(titulo)
        for ax, perp in zip(axes, valores):
            modelo = TSNE_sklearn(
                n_components=2,
                perplexity=perp,
                max_iter=self.__iteraciones,
                learning_rate=self.__tasa_aprendizaje,
                random_state=self.__random_state,
            )
            coords = modelo.fit_transform(self.__datos_escalados)
            ax.scatter(coords[:, 0], coords[:, 1], alpha=0.5, s=10, color='steelblue')
            ax.set_title(f'Perplejidad = {perp}')
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
        plt.tight_layout()

    def __str__(self):
        return ''
