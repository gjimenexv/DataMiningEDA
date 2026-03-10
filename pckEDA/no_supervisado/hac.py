import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist

from pckEDA.eda import AnalisisDatosExploratorio
from .base import NoSupervisado


class HAC(NoSupervisado):
    """Clustering Jerárquico Aglomerativo (HAC) como algoritmo de aprendizaje no supervisado.

    Hereda de NoSupervisado → AnalisisDatosExploratorio, por lo que dispone de todos los
    métodos de carga, limpieza y transformación del EDA antes de ajustar el modelo.

    Flujo de uso:
        1. Crear la instancia con la ruta del CSV.
        2. Aplicar los métodos heredados del EDA (codificarCategorica, eliminarNulos, etc.).
        3. Llamar a ajustar() para entrenar el modelo HAC sobre los datos limpios.
        4. Usar los métodos de visualización para explorar los resultados.
    """

    def __init__(self, path, num, n_clusters=3, metodo='ward', metrica='euclidean'):
        """Carga los datos desde un CSV y configura los parámetros del modelo HAC.

        El modelo no se ajusta en este paso — primero aplica los métodos heredados del EDA
        para limpiar y preparar los datos, luego llama a ajustar().

        Args:
            path: Ruta del archivo CSV a cargar.
            num: Modo de lectura del CSV (1: separador coma con índice, 2: separador punto y coma).
            n_clusters: Número de clusters a formar al cortar el dendrograma. Por defecto 3.
            metodo: Método de enlace para la jerarquía ('ward', 'complete', 'average', 'single').
                    Por defecto 'ward'.
            metrica: Métrica de distancia a utilizar (e.g. 'euclidean', 'cosine').
                     Ignorada cuando metodo='ward' (usa siempre distancia euclídea). Por defecto 'euclidean'.
        """
        AnalisisDatosExploratorio.__init__(self, path, num)
        self.__n_clusters = n_clusters
        self.__metodo = metodo
        self.__metrica = metrica
        self.__datos_escalados = None
        self.__enlace = None
        self.__etiquetas = None
        self.__cophenet_corr = None
        self.__resumen = None

    def ajustar(self):
        """Ajusta el modelo HAC sobre los datos actuales (self.df).

        Debe llamarse después de aplicar los métodos de limpieza y transformación del EDA.
        Estandariza los datos automáticamente con StandardScaler antes de calcular la jerarquía.
        """
        datos = self.df
        escalador = StandardScaler()
        self.__datos_escalados = escalador.fit_transform(datos)
        self.__enlace = linkage(self.__datos_escalados, method=self.__metodo, metric=self.__metrica)
        self.__etiquetas = fcluster(self.__enlace, self.__n_clusters, criterion='maxclust')
        distancias = pdist(self.__datos_escalados)
        self.__cophenet_corr, _ = cophenet(self.__enlace, distancias)
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
    def enlace(self):
        """Devuelve la matriz de enlace generada por scipy (shape: n-1 × 4)."""
        return self.__enlace

    @enlace.setter
    def enlace(self, enlace):
        """Args:
            enlace: Nueva matriz de enlace.
        """
        self.__enlace = enlace

    @property
    def etiquetas(self):
        """Devuelve el array de etiquetas de cluster asignadas a cada observación (1-indexado)."""
        return self.__etiquetas

    @etiquetas.setter
    def etiquetas(self, etiquetas):
        """Args:
            etiquetas: Nuevo array de etiquetas de cluster.
        """
        self.__etiquetas = etiquetas

    @property
    def n_clusters(self):
        """Devuelve el número de clusters formados al cortar el dendrograma."""
        return self.__n_clusters

    @n_clusters.setter
    def n_clusters(self, n_clusters):
        """Args:
            n_clusters: Nuevo número de clusters.
        """
        self.__n_clusters = n_clusters

    @property
    def metodo(self):
        """Devuelve el método de enlace utilizado ('ward', 'complete', 'average', 'single')."""
        return self.__metodo

    @metodo.setter
    def metodo(self, metodo):
        """Args:
            metodo: Nuevo método de enlace.
        """
        self.__metodo = metodo

    @property
    def metrica(self):
        """Devuelve la métrica de distancia utilizada en el cálculo de la jerarquía."""
        return self.__metrica

    @metrica.setter
    def metrica(self, metrica):
        """Args:
            metrica: Nueva métrica de distancia.
        """
        self.__metrica = metrica

    @property
    def cophenet_corr(self):
        """Devuelve el coeficiente de correlación cofenética (0–1).

        Valores cercanos a 1 indican que la jerarquía representa bien las distancias originales.
        """
        return self.__cophenet_corr

    @cophenet_corr.setter
    def cophenet_corr(self, cophenet_corr):
        """Args:
            cophenet_corr: Nuevo coeficiente de correlación cofenética.
        """
        self.__cophenet_corr = cophenet_corr

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

    # ─── Visualization Methods ──────────────────────────────────────────────────

    def plot_dendrograma(self, max_hojas=30, color_umbral=None, titulo='Dendrograma HAC'):
        """Genera el dendrograma de la jerarquía de clusters.

        Muestra la estructura de fusiones del algoritmo HAC, truncado al número
        de hojas especificado para mayor legibilidad con conjuntos grandes.

        Args:
            max_hojas: Número máximo de hojas (observaciones o grupos) a mostrar.
                       Por defecto 30.
            color_umbral: Umbral de distancia para colorear las ramas. Si es None,
                          se utiliza el valor predeterminado de scipy (70% de la
                          distancia máxima). Por defecto None.
            titulo: Título del gráfico. Por defecto 'Dendrograma HAC'.
        """
        plt.style.use('seaborn-v0_8-bright')
        kwargs = dict(
            Z=self.__enlace,
            truncate_mode='lastp',
            p=max_hojas,
            leaf_rotation=90,
            leaf_font_size=10,
            show_contracted=True,
        )
        if color_umbral is not None:
            kwargs['color_threshold'] = color_umbral
        dendrogram(**kwargs)
        plt.title(titulo)
        plt.xlabel('Observaciones (n en paréntesis)')
        plt.ylabel('Distancia de fusión')

    def plot_mapa_calor(self, titulo='Perfil de Clusters'):
        """Genera un mapa de calor con la media de cada variable por cluster.

        Permite identificar visualmente el perfil característico de cada cluster
        (e.g. qué clusters tienen mayor tarifa media, más cancelaciones, etc.).

        Args:
            titulo: Título del gráfico. Por defecto 'Perfil de Clusters'.
        """
        plt.style.use('seaborn-v0_8-bright')
        resumen_norm = (self.__resumen - self.__resumen.mean()) / self.__resumen.std()
        sns.heatmap(
            resumen_norm,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            linewidths=0.5,
            cbar_kws={'label': 'Desviaciones estándar respecto a la media global'},
        )
        plt.title(titulo)
        plt.xlabel('Variable')
        plt.ylabel('Cluster')

    def plot_distribucion(self, titulo='Distribución de Clusters'):
        """Genera un gráfico de barras con el número de observaciones por cluster.

        Permite detectar clusters muy desbalanceados o demasiado pequeños.

        Args:
            titulo: Título del gráfico. Por defecto 'Distribución de Clusters'.
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

        Permite visualizar cómo se separan los clusters en un espacio bidimensional
        definido por dos variables originales del dataset.

        Args:
            col_x: Nombre de la columna a graficar en el eje X.
            col_y: Nombre de la columna a graficar en el eje Y.
            titulo: Título del gráfico. Si es None, se genera automáticamente
                    como 'col_x vs col_y'. Por defecto None.
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
        plt.colorbar(scatter, label='Cluster', ticks=range(1, self.__n_clusters + 1))
        plt.title(titulo)
        plt.xlabel(col_x)
        plt.ylabel(col_y)

    def __str__(self):
        return ''
