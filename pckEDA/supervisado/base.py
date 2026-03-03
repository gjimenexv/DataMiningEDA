from pckEDA.eda import AnalisisDatosExploratorio


class Supervisado(AnalisisDatosExploratorio):
    """Clase base para algoritmos de aprendizaje supervisado.

    Hereda de AnalisisDatosExploratorio y sirve como punto de extensión
    para algoritmos como clasificación y regresión.
    """

    def __init__(self, df):
        """Inicializa la clase con un DataFrame ya cargado.

        Args:
            df: DataFrame de pandas con los datos a analizar.
        """
        self._df = df
