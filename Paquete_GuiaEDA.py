import pandas as pd
import numpy as np
import umap as um
import math
import statistics
pd.options.display.max_rows = 10
import warnings
warnings.filterwarnings('ignore')


class AnalisisDatosExploratorio():
    """Clase para realizar un análisis exploratorio de datos (EDA) en un DataFrame de pandas.
    Permite cargar datos desde un archivo CSV, realizar análisis numérico y completo, y generar gráficos para visualizar la distribución y correlación de los datos.
    Atributos:
        __df (pd.DataFrame): DataFrame que contiene los datos cargados.
        Métodos:
        __init__(self, path, num): Constructor que carga los datos desde un archivo CSV.
        df (property): Propiedad para acceder y modificar el DataFrame.
        analisisNumerico(self): Método para seleccionar solo las columnas numéricas del DataFrame.
        analisisCompleto(self): Método para convertir variables categóricas en variables dummy.
        __cargarDatos(self, path, num): Método privado para cargar datos desde un archivo CSV.
        analisis(self): Método para realizar un análisis completo de los datos, incluyendo estadísticas descriptivas y generación de gráficos.
        __graficosBoxplot(self): Método privado para generar gráficos de boxplot.
        __funcionDensidad(self): Método privado para generar gráficos de función de densidad.
        __histograma(self): Método privado para generar gráficos de histograma.
        __correlaciones(self): Método privado para calcular y mostrar la matriz de correlación.
        __graficoDeCorrelacion(self): Método privado para generar un gráfico de calor de la matriz de correlación."""
    def __init__(self, path, num):
        self.__df = self.__cargarDatos(path, num)
        
    """Propiedad para acceder y modificar el DataFrame."""
    @property
    def df(self):
        return self.__df
    
    """Setter para modificar el DataFrame."""
    @df.setter
    
    def df(self, p_df):
        self.__df = p_df

    """Método para seleccionar solo las columnas numéricas del DataFrame.
    Utiliza el método `select_dtypes()` de pandas para seleccionar solo las columnas que contienen datos numéricos y actualiza el DataFrame con el resultado."""
    def analisisNumerico(self):
        print("Columnas numéricas:", self.__df.select_dtypes(include = ["number"]).columns)
        self.__df = self.__df.select_dtypes(include = ["number"])
    
    """Método para convertir variables categóricas en variables dummy.
    Utiliza la función `pd.get_dummies()` para convertir las variables categóricas en variables dummy (variables binarias) y actualiza el DataFrame con el resultado."""
    def analisisCompleto(self):
        print("Columnas antes de convertir a dummies:", self.__df.columns)
        self.__df = pd.get_dummies(self.__df)
    
    """Método privado para cargar datos desde un archivo CSV.
    Carga los datos desde un archivo CSV utilizando la función `pd.read_csv()`.
    El formato de los datos se determina por el valor del parámetro `num`, que especifica el separador y el formato decimal. 
    Si `num` es 1, se utiliza una coma como separador y un punto como decimal. Si `num` es 2, se utiliza un punto y coma como separador y un punto como decimal. 
    El DataFrame resultante se devuelve al constructor para su almacenamiento en el atributo privado `__df`.
    """
    def __cargarDatos(self, path, num):
        if num == 1:
            return pd.read_csv(path,
            sep = ",",
            decimal = ".",
            index_col = 0)
        if num == 2:
            return pd.read_csv(path,
            sep = ";",
            decimal = ".")
    
    """Método para realizar un análisis completo de los datos, incluyendo estadísticas descriptivas y generación de gráficos."""
    def analisis(self):
        print("Dimensiones:",self.__df.shape)
        print(self.__df.head)
        print("Columnas:", self.__df.columns)
        print(self.__df.describe())
        print("Descripción sin valores nulos:")
        self.__df.dropna().describe()
        print("Media de las columnas numéricas:")
        print(self.__df.mean(numeric_only=True))
        print("Mediana de las columnas numéricas:")
        print(self.__df.median(numeric_only=True))
        print("Desviación estándar de las columnas numéricas:")
        print(self.__df.std(numeric_only=True, ddof = 0))
        print("Valor máximo de las columnas numéricas:")
        print(self.__df.max(numeric_only=True))
        print("Valor mínimo de las columnas numéricas:")
        print(self.__df.min(numeric_only=True))
        print("Cuantiles de las columnas numéricas:")
        print(self.__df.quantile(np.array([0,.33,.50,.75,1]),numeric_only=True))
        self.__graficosBoxplot()
        self.__funcionDensidad()
        self.__histograma()
        self.__correlaciones()
        self.__graficoDeCorrelacion()
    
    """Método privado para generar gráficos de boxplot.
    Crea un gráfico de boxplot para cada columna numérica del DataFrame utilizando el método `boxplot()` de pandas. El gráfico se muestra utilizando `plt.show()`."""
    def __graficosBoxplot(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (15,8), dpi = 200)
        boxplots = self.__df.boxplot(return_type='axes',ax=ax)
        plt.show()
        
    """Método privado para generar gráficos de función de densidad.
    Crea un gráfico de función de densidad para cada columna numérica del DataFrame utilizando el método `plot()` de pandas con el argumento `kind='density'`. El gráfico se muestra utilizando `plt.show()`."""
    def __funcionDensidad(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (12,8), dpi = 200)
        densidad = self.__df[self.__df.columns].plot(kind='density',ax = ax)
        plt.show()

    """Método privado para generar gráficos de histograma.
    Crea un gráfico de histograma para cada columna numérica del DataFrame utilizando el método `plot()` de pandas con el argumento `kind='hist'`. El gráfico se muestra utilizando `plt.show()`.
    """
    def __histograma(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (10,6), dpi = 200)
        histograma = self.__df[self.__df.columns].plot(kind='hist', ax = ax)
        plt.show()

    """Método privado para calcular y mostrar la matriz de correlación.
    Calcula la matriz de correlación utilizando el método `corr()` de pandas, que devuelve un DataFrame con los coeficientes de correlación entre las columnas numéricas del DataFrame original. Luego, imprime la matriz de correlación en la consola.
    """
    def __correlaciones(self):
        corr = self.__df.corr(numeric_only=True)
        print(corr)

    """Método privado para generar un gráfico de calor de la matriz de correlación.
    Crea un gráfico de calor utilizando la biblioteca Seaborn para visualizar la matriz de correlación. El gráfico se muestra utilizando `plt.show()`.
    """
    def __graficoDeCorrelacion(self):
        fig, ax = plt.subplots(figsize=(12, 8), dpi = 150)
        paleta = sns.diverging_palette(220, 10,as_cmap=True).reversed()
        corr = self.__df.corr(numeric_only=True)
        sns.heatmap(corr, vmin= -1, vmax=1, cmap= paleta,square=True, annot=True, ax=ax)
        plt.show()

    
