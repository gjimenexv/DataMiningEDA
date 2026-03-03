from .base import Supervisado


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
