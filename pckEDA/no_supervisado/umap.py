import matplotlib.pyplot as plt
import numpy as np

try:
    from umap import UMAP as UMAP_umap
    _UMAP_DISPONIBLE = True
except ImportError:
    _UMAP_DISPONIBLE = False

from sklearn.preprocessing import StandardScaler

from pckEDA.eda import AnalisisDatosExploratorio
from .base import NoSupervisado


class UMAP(NoSupervisado):
    """Reducción de dimensionalidad con UMAP (Uniform Manifold Approximation and Projection).

    Hereda de NoSupervisado → AnalisisDatosExploratorio, por lo que dispone de todos los
    métodos de carga, limpieza y transformación del EDA antes de ajustar el modelo.

    UMAP proyecta datos de alta dimensionalidad a 2D o 3D preservando tanto la estructura
    local como global de los datos. Es más rápido que t-SNE y soporta proyección de
    nuevos datos mediante transformar().

    Requiere el paquete umap-learn:
        pip install umap-learn

    Flujo de uso:
        1. Crear la instancia con la ruta del CSV.
        2. Aplicar los métodos heredados del EDA (codificarCategorica, eliminarNulos, etc.).
        3. Llamar a ajustar() para calcular la proyección sobre los datos limpios.
        4. Usar los métodos de visualización para explorar la proyección.
    """

    def __init__(self, path, num, n_componentes=2, n_vecinos=15,
                 distancia_minima=0.1, metrica='euclidean', random_state=42):
        """Carga los datos desde un CSV y configura los parámetros del modelo UMAP.

        El modelo no se ajusta en este paso — primero aplica los métodos heredados del EDA
        para limpiar y preparar los datos, luego llama a ajustar().

        Args:
            path: Ruta del archivo CSV a cargar.
            num: Modo de lectura del CSV (1: separador coma con índice, 2: separador punto y coma).
            n_componentes: Dimensiones de la proyección resultante (2 o 3). Por defecto 2.
            n_vecinos: Número de vecinos cercanos a considerar al construir el grafo local.
                       Valores bajos (5–15) resaltan estructura local; valores altos (50–200)
                       preservan mejor la estructura global. Por defecto 15.
            distancia_minima: Distancia mínima permitida entre puntos en el espacio proyectado.
                              Valores bajos (0.0–0.1) producen clusters más compactos;
                              valores altos (0.5–1.0) producen proyecciones más dispersas.
                              Por defecto 0.1.
            metrica: Métrica de distancia para el grafo de vecinos ('euclidean', 'cosine',
                     'manhattan', etc.). Por defecto 'euclidean'.
            random_state: Semilla para reproducibilidad. Por defecto 42.
        """
        if not _UMAP_DISPONIBLE:
            raise ImportError(
                "El paquete 'umap-learn' no está instalado.\n"
                "Instálalo con:  pip install umap-learn"
            )
        AnalisisDatosExploratorio.__init__(self, path, num)
        self.__n_componentes = n_componentes
        self.__n_vecinos = n_vecinos
        self.__distancia_minima = distancia_minima
        self.__metrica = metrica
        self.__random_state = random_state
        self.__datos_escalados = None
        self.__modelo = None
        self.__coordenadas = None

    def ajustar(self):
        """Calcula la proyección UMAP sobre los datos actuales (self.df).

        Debe llamarse después de aplicar los métodos de limpieza y transformación del EDA.
        Estandariza los datos automáticamente con StandardScaler antes de la proyección.
        """
        escalador = StandardScaler()
        self.__datos_escalados = escalador.fit_transform(self.df)

        self.__modelo = UMAP_umap(
            n_components=self.__n_componentes,
            n_neighbors=self.__n_vecinos,
            min_dist=self.__distancia_minima,
            metric=self.__metrica,
            random_state=self.__random_state,
        )
        self.__coordenadas = self.__modelo.fit_transform(self.__datos_escalados)

    def transformar(self, nuevos_datos):
        """Proyecta nuevos datos al espacio UMAP aprendido.

        A diferencia de t-SNE, UMAP soporta proyección de datos no vistos durante el ajuste.

        Args:
            nuevos_datos: Array o DataFrame con los mismos features usados en ajustar().
                          Deben estar ya escalados o en la misma escala que los datos originales.

        Returns:
            Array numpy con las coordenadas proyectadas (n_muestras × n_componentes).
        """
        return self.__modelo.transform(nuevos_datos)

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
        """Devuelve el modelo UMAP ajustado (permite proyectar nuevos datos con transformar())."""
        return self.__modelo

    @modelo.setter
    def modelo(self, modelo):
        """Args:
            modelo: Nuevo modelo UMAP ajustado.
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
    def n_vecinos(self):
        """Devuelve el número de vecinos cercanos utilizado para construir el grafo."""
        return self.__n_vecinos

    @n_vecinos.setter
    def n_vecinos(self, n_vecinos):
        """Args:
            n_vecinos: Nuevo número de vecinos.
        """
        self.__n_vecinos = n_vecinos

    @property
    def distancia_minima(self):
        """Devuelve la distancia mínima entre puntos en el espacio proyectado."""
        return self.__distancia_minima

    @distancia_minima.setter
    def distancia_minima(self, distancia_minima):
        """Args:
            distancia_minima: Nueva distancia mínima.
        """
        self.__distancia_minima = distancia_minima

    @property
    def metrica(self):
        """Devuelve la métrica de distancia utilizada."""
        return self.__metrica

    @metrica.setter
    def metrica(self, metrica):
        """Args:
            metrica: Nueva métrica de distancia.
        """
        self.__metrica = metrica

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

    def plot_proyeccion(self, etiquetas=None, titulo='Proyección UMAP'):
        """Genera un diagrama de dispersión de la proyección 2D UMAP.

        Opcionalmente colorea los puntos con etiquetas externas (e.g. clusters de K-Means o HAC)
        para revelar si la estructura de clustering coincide con la topología del embedding.

        Args:
            etiquetas: Array de etiquetas para colorear los puntos (e.g. hac.etiquetas,
                       kmeans.etiquetas). Si es None, todos los puntos se muestran en el
                       mismo color. Por defecto None.
            titulo: Título del gráfico. Por defecto 'Proyección UMAP'.
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
            plt.scatter(x, y, color='darkorange', alpha=0.6, s=15)
        plt.title(titulo)
        plt.xlabel('UMAP dimensión 1')
        plt.ylabel('UMAP dimensión 2')

    def plot_vecinos(self, valores=(5, 15, 30, 50), titulo='Efecto de n_vecinos — UMAP'):
        """Compara la proyección UMAP con distintos valores de n_vecinos en una cuadrícula.

        Valores bajos resaltan estructura local (clusters pequeños y densos);
        valores altos preservan mejor la estructura global del dataset.

        Args:
            valores: Tupla o lista de valores de n_vecinos a comparar. Por defecto (5, 15, 30, 50).
            titulo: Título general de la figura. Por defecto 'Efecto de n_vecinos — UMAP'.
        """
        plt.style.use('seaborn-v0_8-bright')
        n = len(valores)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        fig.suptitle(titulo)
        for ax, vecinos in zip(axes, valores):
            modelo = UMAP_umap(
                n_components=2,
                n_neighbors=vecinos,
                min_dist=self.__distancia_minima,
                metric=self.__metrica,
                random_state=self.__random_state,
            )
            coords = modelo.fit_transform(self.__datos_escalados)
            ax.scatter(coords[:, 0], coords[:, 1], alpha=0.5, s=10, color='darkorange')
            ax.set_title(f'n_vecinos = {vecinos}')
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
        plt.tight_layout()

    def __str__(self):
        return ''
