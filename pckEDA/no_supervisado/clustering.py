import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil, floor, pi

from .base import NoSupervisado


class Clustering(NoSupervisado):
    """Clase base para algoritmos de clustering.

    Hereda de NoSupervisado → AnalisisDatosExploratorio.
    Centraliza los métodos de visualización compartidos por todos los algoritmos
    de clustering (K-Means, HAC, etc.): mapa de calor, barras de centroides y
    gráfico radar.
    """

    def __init__(self, df, n_clusters):
        """Inicializa la clase con un DataFrame y el número de clusters.

        Args:
            df: DataFrame de pandas con los datos a analizar.
            n_clusters: Número de clusters a formar.
        """
        super().__init__(df)
        self.n_clusters = n_clusters

    # ─── Shared Visualization Methods ──────────────────────────────────────────

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

    def graficarBarrasClusters(self, resumen, titulo='Perfiles de Clusters por Variable',
                               escala=False, clusters=None, variables=None):
        """Genera barras horizontales con el perfil medio de cada cluster.

        Cada subgráfico corresponde a un cluster y muestra el valor medio (o el
        valor escalado) de cada variable. Útil para comparar rápidamente qué
        variables caracterizan a cada grupo.

        Args:
            resumen: DataFrame con la media de cada variable por cluster
                     (filas = clusters, columnas = variables).
            titulo: Título superior del gráfico. Por defecto
                    'Perfiles de Clusters por Variable'.
            escala: Si True, divide cada columna por su valor máximo absoluto
                    para que todas las variables queden en la misma escala relativa.
                    Por defecto False.
            clusters: Lista de índices de posición (0-based) de los clusters a
                      mostrar. Si None, muestra todos. Por defecto None.
            variables: Lista de nombres de columnas a incluir. Si None, incluye
                       todas. Por defecto None.
        """
        plt.style.use('seaborn-v0_8-bright')
        centros = resumen.values.astype(float).copy()
        labels = list(resumen.columns)
        colores = sns.color_palette('Set2', len(labels))

        if variables is not None:
            mask = [c in variables for c in labels]
            centros = centros[:, mask]
            labels = [l for l, m in zip(labels, mask) if m]
            colores = sns.color_palette('Set2', len(labels))

        if escala:
            for col in range(centros.shape[1]):
                col_max = np.abs(centros[:, col]).max()
                if col_max != 0:
                    centros[:, col] /= col_max

        minimo = floor(centros.min()) if floor(centros.min()) < 0 else 0
        maximo = ceil(centros.max())

        cluster_indices = list(range(centros.shape[0])) if clusters is None else clusters
        n = len(cluster_indices)
        altura = max(5, len(labels) * 0.45 + 2)

        fig, axes = plt.subplots(1, n, figsize=(5 * n, altura), dpi=150)
        if n == 1:
            axes = [axes]

        for pos, ci in enumerate(cluster_indices):
            ax = axes[pos]
            valores = centros[ci].tolist()
            ax.barh(range(len(valores)), valores, 2 / 3, color=colores)
            ax.set_xlim(minimo, maximo)
            ax.set_title(f'Cluster {resumen.index[ci]}')
            ax.set_yticks(range(len(labels)))
            if pos == 0:
                ax.set_yticklabels(labels)
            else:
                ax.set_yticklabels([])

        fig.suptitle(titulo, fontsize=13)
        plt.tight_layout()

    def graficarRadarClusters(self, resumen, titulo='Radar de Clusters'):
        """Genera un gráfico radar (araña) con los perfiles normalizados por cluster.

        Normaliza cada variable al rango 0-100% según sus valores mínimo y máximo
        a través de todos los clusters, de modo que las distintas escalas no distorsionen
        la forma del gráfico. Cada cluster se representa con un polígono relleno.

        Nota: con muchas variables (> 15) las etiquetas pueden solaparse; en ese caso
        considera filtrar con ``variables`` o usar ``graficarHeatmapClusters`` en su lugar.

        Args:
            resumen: DataFrame con la media de cada variable por cluster
                     (filas = clusters, columnas = variables).
            titulo: Título del gráfico. Por defecto 'Radar de Clusters'.
        """
        plt.style.use('seaborn-v0_8-bright')
        labels = list(resumen.columns)
        n_vars = len(labels)
        centros_t = resumen.values.astype(float).T  # shape: (n_features, n_clusters)

        centros_norm = np.array([
            ((row - row.min()) / (row.max() - row.min()) * 100)
            if row.max() != row.min() else np.full_like(row, 50.0)
            for row in centros_t
        ])  # shape: (n_features, n_clusters)

        angulos = [n / float(n_vars) * 2 * pi for n in range(n_vars)]
        angulos += angulos[:1]

        fig = plt.figure(figsize=(8, 8), dpi=150)
        ax = fig.add_subplot(111, polar=True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angulos[:-1], labels, size=9)
        ax.set_rlabel_position(0)
        plt.yticks([25, 50, 75, 100], ['25%', '50%', '75%', '100%'],
                   color='grey', size=8)
        plt.ylim(0, 110)

        colores = sns.color_palette('Set2', centros_norm.shape[1])
        for i in range(centros_norm.shape[1]):
            valores = centros_norm[:, i].tolist()
            valores += valores[:1]
            cluster_label = f'Cluster {resumen.index[i]}'
            ax.plot(angulos, valores, linewidth=2, linestyle='solid',
                    label=cluster_label, color=colores[i])
            ax.fill(angulos, valores, alpha=0.15, color=colores[i])

        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title(titulo, size=13, pad=20)
