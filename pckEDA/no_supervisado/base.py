from pckEDA.eda import AnalisisDatosExploratorio


class NoSupervisado(AnalisisDatosExploratorio):
    """Clase base para algoritmos de aprendizaje no supervisado.

    Hereda de AnalisisDatosExploratorio y sirve como punto de extensión
    para algoritmos como clustering y reducción de dimensionalidad.
    """

    def __init__(self, df):
        """Inicializa la clase con un DataFrame ya cargado.

        Args:
            df: DataFrame de pandas con los datos a analizar.
        """
        self._df = df
