from .base import NoSupervisado


class Clustering(NoSupervisado):
    """Clase para realizar clustering en un DataFrame utilizando el algoritmo K-means."""

    def __init__(self, df, n_clusters):
        """Inicializa la clase con un DataFrame y el número de clusters.

        Args:
            df: DataFrame de pandas con los datos a analizar.
            n_clusters: Número de clusters a formar.
        """
        super().__init__(df)
        self.n_clusters = n_clusters
