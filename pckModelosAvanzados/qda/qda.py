"""Semana 6 — Análisis Discriminante Cuadrático (QDA)."""

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from ..base import ModeloBase


class QDA(ModeloBase):
    """Análisis Discriminante Cuadrático.

    Como LDA, pero cada clase tiene su propia matriz de covarianza, lo que da
    fronteras cuadráticas. Se usa cuando las covarianzas difieren entre
    clases; estima más parámetros, así que sobreajusta con pocos datos.
    """

    nombre = "QDA"
    _HIPERPARAMETROS = ("reg_param",)


    def __init__(self, df, target, reg_param=0.0, scaler="standard",
                 random_state=42):
        """Inicializa el QDA.

        Args:
            df: DataFrame con predictoras continuas y el target.
            target: Nombre de la columna objetivo.
            reg_param: Regularización de las covarianzas (0–1). Subirla
                reduce el sobreajuste y estabiliza matrices casi singulares.
            scaler: ``'standard'`` (recomendado), ``'minmax'`` o ``None``.
            random_state: Semilla para el split.
        """
        super().__init__(df, target, scaler, random_state)
        self.__reg_param = reg_param

    # ─── Hiperparámetros ────────────────────────────────────────────────────
    @property
    def reg_param(self):
        """Regularización de las covarianzas (0–1)."""
        return self.__reg_param

    @reg_param.setter
    def reg_param(self, reg_param):
        """Args:
            reg_param: Nuevo valor.
        """
        self.__reg_param = reg_param

    def _crear_modelo(self):
        """Crea el ``QuadraticDiscriminantAnalysis`` con los hiperparámetros actuales."""
        return QuadraticDiscriminantAnalysis(**self.hiperparametros)
