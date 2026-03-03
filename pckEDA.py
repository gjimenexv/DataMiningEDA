import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from prince import PCA as PCA_Prince
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#import umap as um
import math
import statistics

pd.options.display.max_rows = 10
import warnings
warnings.filterwarnings('ignore')


class AnalisisDatosExploratorio():
    """Clase para realizar un análisis exploratorio de datos (EDA) en un DataFrame de pandas."""

    def __init__(self, path, num):
        """Inicializa la clase cargando los datos desde un archivo CSV.

        Args:
            path: Ruta del archivo CSV a cargar.
            num: Modo de lectura del CSV (1: separador coma con índice, 2: separador punto y coma sin índice).
        """
        self.__df = self.__cargarDatos(path, num)

    @property
    def df(self):
        """Propiedad para acceder al DataFrame interno de la clase.

        Returns:
            pd.DataFrame: El DataFrame utilizado para el análisis exploratorio.
        """
        return self.__df

    @df.setter
    def df(self, p_df):
        """Setter para establecer el DataFrame interno de la clase.

        Args:
            p_df: Nuevo DataFrame a asignar.
        """
        self.__df = p_df

    def analisisNumerico(self):
        """Filtra el DataFrame para conservar únicamente las columnas con datos numéricos."""
        self.__df = self.__df.select_dtypes(include = ["number"])

    def analisisCompleto(self):
        """Convierte las variables categóricas del DataFrame en variables dummy (one-hot encoding)."""
        self.__df = pd.get_dummies(self.__df)

    def __cargarDatos(self, path, num):
        """Carga los datos desde un archivo CSV según el modo indicado.

        Args:
            path: Ruta del archivo CSV.
            num: Modo de lectura.
                1 - Separador coma, decimal punto, primera columna como índice.
                2 - Separador punto y coma, decimal punto, sin columna índice.

        Returns:
            pd.DataFrame: DataFrame con los datos cargados.
        """
        if num == 1:
            return pd.read_csv(path,
            sep = ",",
            decimal = ".",
            index_col = 0)
        if num == 2:
            return pd.read_csv(path,
            sep = ";",
            decimal = ".")

    def analisis(self):
        """Realiza un análisis descriptivo completo del DataFrame.

        Imprime las dimensiones, las primeras filas, estadísticas descriptivas
        (media, mediana, desviación estándar, máximo, mínimo y cuantiles)
        y genera gráficos de visualización.
        """
        print("Dimensiones:",self.__df.shape)
        print(self.__df.head)
        print(self.__df.describe())
        self.__df.dropna().describe()
        self.__df.mean(numeric_only=True)
        self.__df.median(numeric_only=True)
        self.__df.std(numeric_only=True, ddof = 0)
        self.__df.max(numeric_only=True)
        self.__df.min(numeric_only=True)
        self.__df.quantile(np.array([0,.33,.50,.75,1]),numeric_only=True)

    def estadisticasDescriptivas(self):
        """Imprime las estadísticas descriptivas de todas las columnas numéricas en dos bloques.

        Bloque 1 - Estadísticas básicas por columna:
            count, mean, std, min, 25%, 50%, 75%, max.

        Bloque 2 - Estadísticas adicionales por columna:
            varianza, asimetría (skewness) y curtosis (kurtosis).
        """
        numericas = self.__df.select_dtypes(include=["number"])

        print("=" * 60)
        print("ESTADÍSTICAS BÁSICAS")
        print("=" * 60)
        print(numericas.describe().T.to_string())

        print("\n" + "=" * 60)
        print("ESTADÍSTICAS ADICIONALES")
        print("=" * 60)
        adicionales = pd.DataFrame({
            "varianza": numericas.var(),
            "asimetría": numericas.skew(),
            "curtosis": numericas.kurt()
        })
        print(adicionales.to_string())

    def graficos(self):
        """Genera un conjunto de gráficos exploratorios: boxplot, densidad, histograma y correlación."""
        self.__graficosBoxplot()
        self.__funcionDensidad()
        self.__histograma()
        self.__correlaciones()
        self.__graficoDeCorrelacion()

    def __graficosBoxplot(self):
        """Genera un gráfico de boxplot para todas las variables del DataFrame."""
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (15,8), dpi = 200)
        boxplots = self.__df.boxplot(return_type='axes',ax=ax)
        plt.show()

    def __funcionDensidad(self):
        """Genera gráficos de densidad (KDE) para cada variable del DataFrame."""
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (15,8), dpi = 200)
        self.__df.plot(kind='density', subplots=True, layout=(4,4), sharex=False, ax=ax)
        plt.show()

    def __histograma(self):
        """Genera histogramas con 20 bins para cada variable del DataFrame."""
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (15,8), dpi = 200)
        self.__df.hist(bins=20, ax=ax)
        plt.show()

    def __correlaciones(self):
        """Calcula e imprime la matriz de correlación de las variables numéricas."""
        print("Matriz de Correlación:")
        corr = self.__df.corr(numeric_only=True)
        print(corr)
        
    def __graficoDeCorrelacion(self):
        """Genera un heatmap de la matriz de correlación usando imshow."""
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (15,8), dpi = 200)
        corr = self.__df.corr(numeric_only=True)
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)
        fig.colorbar(im)
        plt.show()

    def mostrarTamaño(self):
        """Imprime el número de filas y columnas del DataFrame."""
        print("Tamaño del DataFrame:", self.__df.shape)

    def muestraUltimosValores(self, n):
        """Imprime las últimas n filas del DataFrame.

        Args:
            n: Número de filas a mostrar desde el final.
        """
        print("Últimos {} valores: {}".format(n, self.__df.tail(n)))

    def muestraPrimerosValores(self, n):
        """Imprime las primeras n filas del DataFrame.

        Args:
            n: Número de filas a mostrar desde el inicio.
        """
        print("Primeros {} valores: {}".format(n, self.__df.head(n)))

    def muestraTiposDeDatos(self):
        """Imprime los tipos de datos de cada columna del DataFrame."""
        print(self.__df.dtypes)

    def eliminarColumna(self, columna):
        """Elimina una columna del DataFrame.

        Args:
            columna: Nombre de la columna a eliminar.
        """
        self.__df = self.__df.drop(columns=[columna])

    def eliminarColumnas(self, columnas):
        """Elimina múltiples columnas del DataFrame.

        Args:
            columnas: Lista con los nombres de las columnas a eliminar.
        """
        self.__df = self.__df.drop(columns=columnas)

    def eliminarFilas(self, columna, valor):
        """Elimina las filas donde una columna tenga un valor específico.

        Args:
            columna: Nombre de la columna a evaluar.
            valor: Valor que deben tener las filas a eliminar.
        """
        self.__df = self.__df[self.__df[columna] != valor]

    def renombrarColumna(self, columna_antigua, columna_nueva):
        """Renombra una columna del DataFrame.

        Args:
            columna_antigua: Nombre actual de la columna.
            columna_nueva: Nuevo nombre para la columna.
        """
        self.__df = self.__df.rename(columns={columna_antigua: columna_nueva})

    def renombrarColumnas(self, columnas_antiguas, columnas_nuevas):
        """Renombra múltiples columnas del DataFrame.

        Args:
            columnas_antiguas: Lista con los nombres actuales de las columnas.
            columnas_nuevas: Lista con los nuevos nombres correspondientes.
        """
        self.__df = self.__df.rename(columns=dict(zip(columnas_antiguas, columnas_nuevas)))

    def codificarCategorica(self, columna, mapeo=None):
        """Transforma una columna categórica a valores numéricos (label encoding o mapeo personalizado).

        Existen dos modos de uso:

        1. Label encoding automático (mapeo=None):
           Asigna un entero a cada categoría única en orden alfabético,
           comenzando desde 0. Útil cuando las categorías no tienen un
           orden o jerarquía significativa.
           Ejemplo: ["City Hotel", "Resort Hotel"] → [0, 1]

        2. Mapeo personalizado / enum (mapeo={...}):
           Aplica la conversión definida por el usuario mediante un
           diccionario {categoria: valor_numerico}. Útil cuando las
           categorías tienen un orden lógico o jerarquía definida
           (encoding ordinal), o cuando se requieren valores específicos
           para cada categoría (tipo enum).
           Ejemplo: {"No Deposit": 0, "Non Refund": 1, "Refundable": 2}

        Args:
            columna: Nombre de la columna categórica a transformar.
            mapeo: Diccionario opcional con la correspondencia {categoria: valor_numerico}.
                   Si es None, se aplica label encoding automático.

        Raises:
            ValueError: Si la columna no existe en el DataFrame.

        Examples:
            # Label encoding automático
            eda.codificarCategorica("hotel")

            # Mapeo personalizado (ordinal / enum)
            eda.codificarCategorica("deposit_type", mapeo={"No Deposit": 0, "Non Refund": 1, "Refundable": 2})
        """
        if columna not in self.__df.columns:
            raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

        if mapeo is None:
            categorias = sorted(self.__df[columna].dropna().unique())
            mapeo = {cat: i for i, cat in enumerate(categorias)}

        self.__df[columna] = self.__df[columna].map(mapeo)
        print(f"Columna '{columna}' codificada con el mapeo:")
        for cat, val in mapeo.items():
            print(f"  {cat} → {val}")

    def eliminarDuplicados(self):
        """Elimina las filas duplicadas del DataFrame."""
        antes = self.__df.shape[0]
        self.__df = self.__df.drop_duplicates()
        despues = self.__df.shape[0]
        print(f"Se eliminaron {antes - despues} filas duplicadas. Total actual: {despues} filas.")

    def eliminarNulos(self):
        """Elimina las filas con valores nulos del DataFrame.

        Imprime la cantidad de nulos por columna antes y después de la eliminación.
        """
        nulos_antes = self.__df.isnull().sum().sum()
        filas_antes = self.__df.shape[0]
        print(f"Cantidad total de valores nulos antes de eliminar: {nulos_antes} en {filas_antes} filas.")
        print(self.__df.isnull().sum())
        self.__df = self.__df.dropna()
        self.__df.count()
        nulos_despues = self.__df.isnull().sum().sum()
        print(f"Cantidad total de valores nulos después de eliminar: {nulos_despues} en {self.__df.shape[0]} filas.")

    def reemplazarNulos(self, columna, metodo="mean"):
        """Reemplaza los valores nulos de una columna con la media o la mediana.

        Args:
            columna: Nombre de la columna a procesar.
            metodo: Método de imputación, "mean" para media o "median" para mediana.
                    Por defecto usa "mean".

        Raises:
            ValueError: Si el método indicado no es "mean" ni "median".
        """
        if metodo == "mean":
            valor = self.__df[columna].mean()
        elif metodo == "median":
            valor = self.__df[columna].median()
        else:
            raise ValueError("Método no válido. Usa 'mean' o 'median'.")

        nulos_antes = self.__df[columna].isnull().sum()
        self.__df[columna] = self.__df[columna].fillna(valor)
        print(f"Columna '{columna}': {nulos_antes} nulos reemplazados con {metodo} ({valor:.2f})")

    def detectorDeOutliers(self, columna):
        """Detecta e imprime los outliers de una columna usando el método IQR.

        Args:
            columna: Nombre de la columna a analizar.
        """
        Q1 = self.__df[columna].quantile(0.25)
        Q3 = self.__df[columna].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.__df[(self.__df[columna] < lower_bound) | (self.__df[columna] > upper_bound)]
        print("Outliers en la columna '{}':".format(columna))
        print(outliers)

    def eliminarOutliers(self, columna):
        """Elimina los outliers de una columna usando el método IQR (1.5 * IQR).

        Args:
            columna: Nombre de la columna de la cual eliminar outliers.
        """
        Q1 = self.__df[columna].quantile(0.25)
        Q3 = self.__df[columna].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        self.__df = self.__df[(self.__df[columna] >= lower_bound) & (self.__df[columna] <= upper_bound)]

    def graficarFrecuencias(self, title, xlabel, ylabel, columna):
        """Genera un gráfico de barras con las 40 categorías más frecuentes de la columna parametro.

        Args:
            title: Título del gráfico.
            xlabel: Etiqueta del eje X.
            ylabel: Etiqueta del eje Y.
        """
        self.__df[columna].value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.show()

    def graficarHeatmap(self, title):
        """Genera un heatmap de correlación usando seaborn con anotaciones numéricas.

        Args:
            title: Título del gráfico.
            
        """
        corr = self.__df.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
        plt.title(title + " - Heatmap de Correlación")
        plt.show()

    def graficarScatter(self, x_col, y_col):
        """Genera un gráfico de dispersión (scatter plot) entre dos columnas.

        Args:
            x_col: Nombre de la columna para el eje X.
            y_col: Nombre de la columna para el eje Y.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.__df, x=x_col, y=y_col)
        plt.title(f"Scatter Plot of {x_col} vs {y_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.show()

    def graficarBoxplot(self, column):
        """Genera un boxplot para una columna específica del DataFrame.

        Args:
            column: Nombre de la columna a graficar.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.__df[column])
        plt.title(f"Boxplot of {column}")
        plt.xlabel(column)
        plt.show()
    
    def graficarBoxplotComparativo(self, x_col, y_col):
        """Genera un boxplot comparativo entre dos columnas del DataFrame.

        Args:
            x_col: Nombre de la columna para el eje X (categoría).
            y_col: Nombre de la columna para el eje Y (valores numéricos).
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.__df[x_col], y=self.__df[y_col])
        plt.title(f"Boxplot Comparativo de {y_col} por {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.xticks(rotation=45)
        plt.show()

    def graficarBoxplotTodasColumnas(self):
        """Genera un boxplot individual para cada columna numérica del DataFrame."""
        numeric_cols = self.__df.select_dtypes(include='number').columns
        for col in numeric_cols:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=self.__df[col])
            plt.title(f"Boxplot de {col}")
            plt.xlabel(col)
            plt.show()
            
# Boxplot
    def graficoBoxplotOriginal(self):
        """Genera un gráfico de boxplot para cada columna numérica del DataFrame.
        Organiza los gráficos en una cuadrícula de 3 columnas y filas suficientes para mostrar todas las variables numéricas.
        """
        columnas_numericas = self.__df.select_dtypes(include='number').columns
        n = len(columnas_numericas)
        columnas = 3
        filas = math.ceil(n / columnas)
        fig, axes = plt.subplots(filas, columnas, figsize=(5 * columnas, 4 * filas), dpi=158)
        axes = axes.flatten()
        colores = sns.color_palette("Set3", n)
        for i, col in enumerate(columnas_numericas):
            sns.boxplot(y=self.__df[col], ax=axes[i], color=colores[i])
            axes[i].set_title(f"Boxplot de {col}", fontsize=10)
            axes[i].set_ylabel(col)
            axes[i].grid(True, linestyle='-', alpha=0.5)
        for j in range(1 + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

#NoSupervisado

class NoSupervisado(AnalisisDatosExploratorio):
    """Clase para realizar análisis de datos no supervisados, como clustering y reducción de dimensionalidad."""

    def __init__(self, df):
        """Inicializa la clase con un DataFrame.

        Args:
            df: DataFrame de pandas con los datos a analizar.
        """
        self.__df = df

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


class ACP(NoSupervisado):
    """Análisis de Componentes Principales (ACP) como algoritmo no supervisado de reducción de dimensionalidad.

    Hereda de NoSupervisado y utiliza la biblioteca Prince para ajustar el modelo PCA,
    calculando coordenadas, correlaciones, contribuciones y calidad de representación (cos²)
    de los individuos y variables en el espacio reducido.
    """

    def __init__(self, datos, n_componentes=5):
        """Inicializa el modelo ACP ajustándolo sobre los datos proporcionados.

        Args:
            datos: DataFrame de pandas con los datos numéricos a analizar.
            n_componentes: Número de componentes principales a retener. Por defecto 5.
        """
        super().__init__(datos)
        self.__datos = datos
        self.__modelo = PCA_Prince(n_components=n_componentes).fit(self.__datos)
        self.__correlacion_var = self.__modelo.column_correlations
        self.__coordenadas_ind = self.__modelo.row_coordinates(self.__datos)
        self.__contribucion_ind = self.__modelo.row_contributions_
        self.__cos2_ind = self.__modelo.row_cosine_similarities(self.__datos)
        self.__var_explicada = self.__modelo.percentage_of_variance_

    @property
    def datos(self):
        """Devuelve los datos utilizados para el análisis."""
        return self.__datos

    @datos.setter
    def datos(self, datos):
        """Permite actualizar los datos.

        Args:
            datos: Nuevo DataFrame a asignar.
        """
        self.__datos = datos

    @property
    def modelo(self):
        """Devuelve el modelo PCA ajustado."""
        return self.__modelo

    @modelo.setter
    def modelo(self, modelo):
        """Permite actualizar el modelo PCA ajustado.

        Args:
            modelo: Nuevo modelo PCA_Prince ajustado.
        """
        self.__modelo = modelo

    @property
    def correlacion_var(self):
        """Devuelve la correlación de las variables originales con cada componente principal."""
        return self.__correlacion_var

    @correlacion_var.setter
    def correlacion_var(self, correlacion_var):
        """Args:
            correlacion_var: Nueva matriz de correlación variables-componentes.
        """
        self.__correlacion_var = correlacion_var

    @property
    def coordenadas_ind(self):
        """Devuelve las coordenadas de los individuos en el espacio de componentes principales."""
        return self.__coordenadas_ind

    @coordenadas_ind.setter
    def coordenadas_ind(self, coordenadas_ind):
        """Args:
            coordenadas_ind: Nuevas coordenadas de los individuos.
        """
        self.__coordenadas_ind = coordenadas_ind

    @property
    def contribucion_ind(self):
        """Devuelve la contribución de cada individuo a los componentes principales."""
        return self.__contribucion_ind

    @contribucion_ind.setter
    def contribucion_ind(self, contribucion_ind):
        """Args:
            contribucion_ind: Nueva contribución de los individuos.
        """
        self.__contribucion_ind = contribucion_ind

    @property
    def cos2_ind(self):
        """Devuelve la calidad de representación (cos²) de cada individuo en el plano principal."""
        return self.__cos2_ind

    @cos2_ind.setter
    def cos2_ind(self, cos2_ind):
        """Args:
            cos2_ind: Nuevo cos² de los individuos.
        """
        self.__cos2_ind = cos2_ind

    @property
    def var_explicada(self):
        """Devuelve el porcentaje de varianza explicada por cada componente principal."""
        return self.__var_explicada

    @var_explicada.setter
    def var_explicada(self, var_explicada):
        """Args:
            var_explicada: Nuevo vector de varianza explicada.
        """
        self.__var_explicada = var_explicada

    def plot_plano_principal(self, ejes=[0, 1], ind_labels=True, titulo='Plano Principal'):
        """Genera un gráfico de dispersión de los individuos en el plano principal.

        Args:
            ejes: Lista de dos enteros indicando los componentes a graficar. Por defecto [0, 1].
            ind_labels: Si True, muestra las etiquetas de los individuos. Por defecto True.
            titulo: Título del gráfico.
        """
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values
        plt.style.use('seaborn-v0_8-bright')
        plt.scatter(x, y, color='gray')
        plt.title(titulo)
        plt.axhline(y=0, color='dimgrey', linestyle='--')
        plt.axvline(x=0, color='dimgrey', linestyle='--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))

    def plot_circulo(self, ejes=[0, 1], var_labels=True, titulo='Círculo de Correlación'):
        """Genera el círculo de correlación de las variables con los componentes principales.

        Args:
            ejes: Lista de dos enteros indicando los componentes a graficar. Por defecto [0, 1].
            var_labels: Si True, muestra las etiquetas de las variables. Por defecto True.
            titulo: Título del gráfico.
        """
        cor = self.correlacion_var.iloc[:, ejes].values
        plt.style.use('seaborn-v0_8-bright')
        c = plt.Circle((0, 0), radius=1, color='steelblue', fill=False)
        plt.gca().add_patch(c)
        plt.axis('scaled')
        plt.title(titulo)
        plt.axhline(y=0, color='dimgrey', linestyle='--')
        plt.axvline(x=0, color='dimgrey', linestyle='--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * 0.95, cor[i, 1] * 0.95, color='steelblue',
                      alpha=0.5, head_width=0.05, head_length=0.05)
            if var_labels:
                plt.text(cor[i, 0] * 1.05, cor[i, 1] * 1.05, self.correlacion_var.index[i],
                         color='steelblue', ha='center', va='center')

    def plot_sobreposicion(self, ejes=[0, 1], ind_labels=True,
                           var_labels=True, titulo='Sobreposición Plano-Círculo'):
        """Genera un gráfico de sobreposición entre el plano principal y el círculo de correlación.

        Args:
            ejes: Lista de dos enteros indicando los componentes a graficar. Por defecto [0, 1].
            ind_labels: Si True, muestra las etiquetas de los individuos. Por defecto True.
            var_labels: Si True, muestra las etiquetas de las variables. Por defecto True.
            titulo: Título del gráfico.
        """
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values
        cor = self.correlacion_var.iloc[:, ejes]
        scale = min((max(x) - min(x) / (max(cor[ejes[0]]) - min(cor[ejes[0]]))),
                    (max(y) - min(y) / (max(cor[ejes[1]]) - min(cor[ejes[1]])))) * 0.7
        cor = self.correlacion_var.iloc[:, ejes].values
        plt.style.use('seaborn-v0_8-bright')
        plt.axhline(y=0, color='dimgrey', linestyle='--')
        plt.axvline(x=0, color='dimgrey', linestyle='--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        plt.scatter(x, y, color='gray')
        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * scale, cor[i, 1] * scale, color='steelblue',
                      alpha=0.5, head_width=0.05, head_length=0.05)
            if var_labels:
                plt.text(cor[i, 0] * scale * 1.15, cor[i, 1] * scale * 1.15,
                         self.correlacion_var.index[i],
                         color='steelblue', ha='center', va='center')

    def __str__(self):
        return ''


#Supervisado
class Supervisado(AnalisisDatosExploratorio):
    """Clase para realizar análisis de datos supervisados, como clasificación y regresión."""

    def __init__(self, df):
        """Inicializa la clase con un DataFrame.

        Args:
            df: DataFrame de pandas con los datos a analizar.
        """
        super().__init__(df)
        self.__df = df
        
class Clasificacion(Supervisado):
    """Clase para realizar clasificación en un DataFrame utilizando algoritmos de machine learning."""

    def __init__(self, df, target):
        """Inicializa la clase con un DataFrame y la variable objetivo.

        Args:
            df: DataFrame de pandas con los datos a analizar.
            target: Nombre de la columna que contiene la variable objetivo (clase).
        """
        super().__init__(df)
        self.target = target
        
class Regresion(Supervisado):
    """Clase para realizar regresión en un DataFrame utilizando algoritmos de machine learning."""

    def __init__(self, df, target):
        """Inicializa la clase con un DataFrame y la variable objetivo.

        Args:
            df: DataFrame de pandas con los datos a analizar.
            target: Nombre de la columna que contiene la variable objetivo (valor a predecir).
        """
        super().__init__(df)
        self.target = target