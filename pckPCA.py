import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prince import PCA as PCA_Prince
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

class ACP:
    
    def __init__(self, datos, n_componentes = 5):
        """Clase para realizar Análisis de Componentes Principales (ACP) utilizando la biblioteca Prince.
        Args:
            datos (pd.DataFrame): Datos a analizar.
            n_componentes (int): Número de componentes principales a retener.
        """
        self.__datos = datos
        self.__modelo = PCA_Prince(n_components = n_componentes).fit(self.__datos)
        self.__correlacion_var = self.__modelo.column_correlations
        self.__coordenadas_ind = self.__modelo.row_coordinates(self.__datos)
        self.__contribucion_ind = self.__modelo.row_contributions_
        self.__cos2_ind = self.__modelo.row_cosine_similarities(self.__datos)
        self.__var_explicada = self.__modelo.percentage_of_variance_
        
    @property
    def datos(self):
        """Devuelve los datos utilizados para el análisis.
        """
        return self.__datos
    @datos.setter
    def datos(self, datos):
        """Permite actualizar los datos y recalcular el modelo.
        Args:
            datos (pd.DataFrame): Nuevos datos a analizar.
        """
        self.__datos = datos
        
    @property
    def modelo(self):
        return self.__modelo
    @property
    def correlacion_var(self):
        return self.__correlacion_var
    @property
    def coordenadas_ind(self):
        return self.__coordenadas_ind
    @property
    def contribucion_ind(self):
        return self.__contribucion_ind
    @property
    def cos2_ind(self):
        return self.__cos2_ind
    @property
    def var_explicada(self):
        return self.__var_explicada
    @var_explicada.setter
    def var_explicada(self, var_explicada):
        self.__var_explicada = var_explicada
    @modelo.setter
    def modelo(self, modelo):
        self.__modelo = modelo
    @correlacion_var.setter
    def correlacion_var(self, correlacion_var):
        self.__correlacion_var = correlacion_var
    @coordenadas_ind.setter
    def coordenadas_ind(self, coordenadas_ind):
        self.__coordenadas_ind = coordenadas_ind
    @contribucion_ind.setter
    def contribucion_ind(self, contribucion_ind):
        self.__contribucion_ind = contribucion_ind
    @cos2_ind.setter
    def cos2_ind(self, cos2_ind):
        self.__cos2_ind = cos2_ind


    def plot_plano_principal(self, ejes = [0, 1], ind_labels = True, titulo = 'Plano Principal'):
        """Genera un gráfico de dispersión de los individuos en el plano definido por los ejes seleccionados.
        Args:
            ejes (list): Lista de dos enteros que indican los componentes a graficar (por ejemplo, [0, 1] para el primer y segundo componente).
            ind_labels (bool): Indica si se deben mostrar las etiquetas de los individuos en el gráfico.
            titulo (str): Título del gráfico.
        """
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values
        plt.style.use('seaborn-v0_8-bright')
        plt.scatter(x, y, color = 'gray')
        plt.title(titulo)
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))
                
                
    def plot_circulo(self, ejes = [0, 1], var_labels = True, titulo = 'Círculo de Correlación'):
        """Genera un gráfico del círculo de correlación para los variables en el plano definido por los ejes seleccionados.
        Args:
            ejes (list): Lista de dos enteros que indican los componentes a graficar (por ejemplo, [0, 1] para el primer y segundo componente).
            var_labels (bool): Indica si se deben mostrar las etiquetas de las variables en el gráfico.
            titulo (str): Título del gráfico.
        """
        cor = self.correlacion_var.iloc[:, ejes].values
        plt.style.use('seaborn-v0_8-bright')
        c = plt.Circle((0, 0), radius = 1, color = 'steelblue', fill = False)
        plt.gca().add_patch(c)
        plt.axis('scaled')
        plt.title(titulo)
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * 0.95, cor[i, 1] * 0.95, color = 'steelblue', 
                      alpha = 0.5, head_width = 0.05, head_length = 0.05)
            if var_labels:
                plt.text(cor[i, 0] * 1.05, cor[i, 1] * 1.05, self.correlacion_var.index[i], 
                         color = 'steelblue', ha = 'center', va = 'center')
                
    def plot_sobreposicion(self, ejes = [0, 1], ind_labels = True,
                      var_labels = True, titulo = 'Sobreposición Plano-Círculo'):
        """Genera un gráfico de sobreposición entre el plano de los individuos y el círculo de correlación.
        Args:
            ejes (list): Lista de dos enteros que indican los componentes a graficar (por ejemplo, [0, 1] para el primer y segundo componente).
            ind_labels (bool): Indica si se deben mostrar las etiquetas de los individuos en el gráfico.
            var_labels (bool): Indica si se deben mostrar las etiquetas de las variables en el gráfico.
            titulo (str): Título del gráfico.
        """
        
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values
        cor = self.correlacion_var.iloc[:, ejes]
        scale = min((max(x) - min(x)/(max(cor[ejes[0]]) - min(cor[ejes[0]]))), 
                    (max(y) - min(y)/(max(cor[ejes[1]]) - min(cor[ejes[1]])))) * 0.7
        cor = self.correlacion_var.iloc[:, ejes].values
        plt.style.use('seaborn-v0_8-bright')
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        plt.scatter(x, y, color = 'gray')
        
        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * scale, cor[i, 1] * scale, color = 'steelblue', 
                      alpha = 0.5, head_width = 0.05, head_length = 0.05)
            if var_labels:
                plt.text(cor[i, 0] * scale * 1.15, cor[i, 1] * scale * 1.15, 
                         self.correlacion_var.index[i], 
                         color = 'steelblue', ha = 'center', va = 'center')

    def __str__(self):
       return ''