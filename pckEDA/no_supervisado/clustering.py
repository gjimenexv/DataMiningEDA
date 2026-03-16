import matplotlib.pyplot as plt
import seaborn as sns

from .base import NoSupervisado


class Clustering(NoSupervisado):
    """Clase base para algoritmos de clustering.

    Hereda de NoSupervisado → AnalisisDatosExploratorio.
    Añade el método de visualización de perfiles de clusters compartido por
    todos los algoritmos de clustering (K-Means, HAC, etc.).
    """

    def __init__(self, df, n_clusters):
        """Inicializa la clase con un DataFrame y el número de clusters.

        Args:
            df: DataFrame de pandas con los datos a analizar.
            n_clusters: Número de clusters a formar.
        """
        super().__init__(df)
        self.n_clusters = n_clusters

    def graficarHeatmapClusters(self, resumen, titulo='Perfil de Clusters'):
        """Genera un mapa de calor con la media normalizada de cada variable por cluster.

        Normaliza las medias por columna (z-score) para que variables con distintas
        escalas sean comparables. Valores positivos (verde) indican que el cluster
        está por encima de la media global; negativos (rojo), por debajo.

        Args:
            resumen: DataFrame con la media de cada variable por cluster
                     (filas = clusters, columnas = variables).
            titulo: Título del gráfico. Por defecto 'Perfil de Clusters'.
        """
        plt.style.use('seaborn-v0_8-bright')
        resumen_norm = (resumen - resumen.mean()) / resumen.std()
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
