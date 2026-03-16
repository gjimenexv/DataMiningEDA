import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans as KMeans_sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from pckEDA.eda import AnalisisDatosExploratorio
from .clustering import Clustering


class KMeans(Clustering):
    """Clustering por K-Means como algoritmo de aprendizaje no supervisado.

    Hereda de NoSupervisado → AnalisisDatosExploratorio, por lo que dispone de todos los
    métodos de carga, limpieza y transformación del EDA antes de ajustar el modelo.

    K-Means divide los datos en k grupos minimizando la varianza intra-cluster
    (suma de distancias cuadradas al centroide). Los datos se estandarizan automáticamente.

    Flujo de uso:
        1. Crear la instancia con la ruta del CSV.
        2. Aplicar los métodos heredados del EDA (codificarCategorica, eliminarNulos, etc.).
        3. Llamar a ajustar() para entrenar el modelo K-Means sobre los datos limpios.
        4. Usar los métodos de visualización para explorar los resultados.
    """

    def __init__(self, path, num, n_clusters=3, init='k-means++',
                 n_init=10, random_state=42):
        """Carga los datos desde un CSV y configura los parámetros del modelo K-Means.

        El modelo no se ajusta en este paso — primero aplica los métodos heredados del EDA
        para limpiar y preparar los datos, luego llama a ajustar().

        Args:
            path: Ruta del archivo CSV a cargar.
            num: Modo de lectura del CSV (1: separador coma con índice, 2: separador punto y coma).
            n_clusters: Número de clusters a formar. Por defecto 3.
            init: Método de inicialización de centroides ('k-means++' o 'random').
                  'k-means++' produce resultados más estables. Por defecto 'k-means++'.
            n_init: Número de ejecuciones con distintas semillas. Se conserva la mejor.
                    Por defecto 10.
            random_state: Semilla para reproducibilidad. Por defecto 42.
        """
        AnalisisDatosExploratorio.__init__(self, path, num)
        self.__n_clusters = n_clusters
        self.__init_method = init
        self.__n_init = n_init
        self.__random_state = random_state
        self.__datos_escalados = None
        self.__modelo = None
        self.__etiquetas = None
        self.__centroides = None
        self.__inercia = None
        self.__silhouette = None
        self.__resumen = None

    def ajustar(self):
        """Ajusta el modelo K-Means sobre los datos actuales (self.df).

        Debe llamarse después de aplicar los métodos de limpieza y transformación del EDA.
        Estandariza los datos automáticamente con StandardScaler antes del ajuste.
        """
        datos = self.df
        escalador = StandardScaler()
        self.__datos_escalados = escalador.fit_transform(datos)

        self.__modelo = KMeans_sklearn(
            n_clusters=self.__n_clusters,
            init=self.__init_method,
            n_init=self.__n_init,
            random_state=self.__random_state,
        ).fit(self.__datos_escalados)

        self.__etiquetas = self.__modelo.labels_
        self.__centroides = self.__modelo.cluster_centers_
        self.__inercia = self.__modelo.inertia_
        self.__silhouette = silhouette_score(self.__datos_escalados, self.__etiquetas)

        df_resumen = datos.copy()
        df_resumen['cluster'] = self.__etiquetas
        self.__resumen = df_resumen.groupby('cluster').mean()

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
        """Devuelve el modelo KMeans de scikit-learn ajustado."""
        return self.__modelo

    @modelo.setter
    def modelo(self, modelo):
        """Args:
            modelo: Nuevo modelo KMeans ajustado.
        """
        self.__modelo = modelo

    @property
    def etiquetas(self):
        """Devuelve el array de etiquetas de cluster asignadas a cada observación (0-indexado)."""
        return self.__etiquetas

    @etiquetas.setter
    def etiquetas(self, etiquetas):
        """Args:
            etiquetas: Nuevo array de etiquetas de cluster.
        """
        self.__etiquetas = etiquetas

    @property
    def centroides(self):
        """Devuelve la matriz de centroides en el espacio escalado (n_clusters × n_features)."""
        return self.__centroides

    @centroides.setter
    def centroides(self, centroides):
        """Args:
            centroides: Nueva matriz de centroides.
        """
        self.__centroides = centroides

    @property
    def inercia(self):
        """Devuelve la inercia (suma de distancias cuadradas al centroide más cercano).

        Valores más bajos indican clusters más compactos.
        """
        return self.__inercia

    @inercia.setter
    def inercia(self, inercia):
        """Args:
            inercia: Nueva inercia del modelo.
        """
        self.__inercia = inercia

    @property
    def silhouette(self):
        """Devuelve el coeficiente de Silhouette promedio (-1 a 1).

        Valores cercanos a 1 indican clusters bien separados y compactos.
        Valores cercanos a 0 indican clusters solapados.
        Valores negativos indican asignaciones incorrectas.
        """
        return self.__silhouette

    @silhouette.setter
    def silhouette(self, silhouette):
        """Args:
            silhouette: Nuevo coeficiente de Silhouette.
        """
        self.__silhouette = silhouette

    @property
    def resumen(self):
        """Devuelve un DataFrame con la media de cada variable por cluster."""
        return self.__resumen

    @resumen.setter
    def resumen(self, resumen):
        """Args:
            resumen: Nuevo DataFrame de resumen por cluster.
        """
        self.__resumen = resumen

    @property
    def n_clusters(self):
        """Devuelve el número de clusters configurado."""
        return self.__n_clusters

    @n_clusters.setter
    def n_clusters(self, n_clusters):
        """Args:
            n_clusters: Nuevo número de clusters.
        """
        self.__n_clusters = n_clusters

    @property
    def init_method(self):
        """Devuelve el método de inicialización de centroides utilizado."""
        return self.__init_method

    @init_method.setter
    def init_method(self, init_method):
        """Args:
            init_method: Nuevo método de inicialización.
        """
        self.__init_method = init_method

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

    def plot_codo(self, max_clusters=10, titulo='Método del Codo — K-Means'):
        """Genera el gráfico del método del codo para determinar el número óptimo de clusters.

        Entrena K-Means para cada k entre 2 y max_clusters y grafica la inercia resultante.
        El punto de inflexión ("codo") sugiere el valor óptimo de k.

        Args:
            max_clusters: Número máximo de clusters a evaluar. Por defecto 10.
            titulo: Título del gráfico. Por defecto 'Método del Codo — K-Means'.
        """
        plt.style.use('seaborn-v0_8-bright')
        inercias = []
        rango = range(2, max_clusters + 1)
        for k in rango:
            km = KMeans_sklearn(
                n_clusters=k,
                init=self.__init_method,
                n_init=self.__n_init,
                random_state=self.__random_state,
            ).fit(self.__datos_escalados)
            inercias.append(km.inertia_)
        plt.plot(rango, inercias, marker='o', color='steelblue')
        plt.axvline(x=self.__n_clusters, color='tomato', linestyle='--',
                    label=f'k actual = {self.__n_clusters}')
        plt.title(titulo)
        plt.xlabel('Número de clusters (k)')
        plt.ylabel('Inercia')
        plt.legend()

    def plot_silhouette(self, max_clusters=10, titulo='Coeficiente de Silhouette por k'):
        """Genera el gráfico del coeficiente de Silhouette para distintos valores de k.

        Permite identificar el número de clusters con mejor separación interna.
        El valor de k con mayor Silhouette es el óptimo.

        Args:
            max_clusters: Número máximo de clusters a evaluar. Por defecto 10.
            titulo: Título del gráfico. Por defecto 'Coeficiente de Silhouette por k'.
        """
        plt.style.use('seaborn-v0_8-bright')
        scores = []
        rango = range(2, max_clusters + 1)
        for k in rango:
            km = KMeans_sklearn(
                n_clusters=k,
                init=self.__init_method,
                n_init=self.__n_init,
                random_state=self.__random_state,
            ).fit(self.__datos_escalados)
            scores.append(silhouette_score(self.__datos_escalados, km.labels_))
        plt.plot(rango, scores, marker='o', color='seagreen')
        plt.axvline(x=self.__n_clusters, color='tomato', linestyle='--',
                    label=f'k actual = {self.__n_clusters}')
        plt.title(titulo)
        plt.xlabel('Número de clusters (k)')
        plt.ylabel('Silhouette')
        plt.legend()

    def plot_mapa_calor(self, titulo='Perfil de Clusters — K-Means'):
        """Genera un mapa de calor con la media de cada variable por cluster.

        Delega en el método heredado graficarHeatmapClusters() de AnalisisDatosExploratorio.

        Args:
            titulo: Título del gráfico. Por defecto 'Perfil de Clusters — K-Means'.
        """
        self.graficarHeatmapClusters(self.__resumen, titulo)

    def plot_distribucion(self, titulo='Distribución de Clusters — K-Means'):
        """Genera un gráfico de barras con el número de observaciones por cluster.

        Args:
            titulo: Título del gráfico. Por defecto 'Distribución de Clusters — K-Means'.
        """
        plt.style.use('seaborn-v0_8-bright')
        conteos = pd.Series(self.__etiquetas).value_counts().sort_index()
        colores = plt.cm.tab10.colors[:len(conteos)]
        plt.bar(conteos.index.astype(str), conteos.values, color=colores)
        plt.title(titulo)
        plt.xlabel('Cluster')
        plt.ylabel('Número de observaciones')
        for i, (etiqueta, valor) in enumerate(conteos.items()):
            plt.text(i, valor + max(conteos) * 0.01, str(valor), ha='center', va='bottom')

    def plot_dispersion(self, col_x, col_y, titulo=None):
        """Genera un diagrama de dispersión de dos variables coloreado por cluster.

        Args:
            col_x: Nombre de la columna a graficar en el eje X.
            col_y: Nombre de la columna a graficar en el eje Y.
            titulo: Título del gráfico. Si es None, se genera automáticamente. Por defecto None.
        """
        plt.style.use('seaborn-v0_8-bright')
        if titulo is None:
            titulo = f'{col_x} vs {col_y} por Cluster'
        cmap = plt.cm.get_cmap('tab10', self.__n_clusters)
        scatter = plt.scatter(
            self.df[col_x],
            self.df[col_y],
            c=self.__etiquetas,
            cmap=cmap,
            alpha=0.6,
            s=20,
        )
        plt.colorbar(scatter, label='Cluster', ticks=range(self.__n_clusters))
        plt.title(titulo)
        plt.xlabel(col_x)
        plt.ylabel(col_y)

    def __str__(self):
        return ''
