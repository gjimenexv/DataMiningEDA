"""Comparación de varios modelos del paquete en una sola tabla."""

import time

import pandas as pd


def comparar_modelos(modelos, cv=5, scoring="f1_macro"):
    """Ajusta y compara varios modelos con el mismo split de datos.

    Replica la comparación de las tareas (S4) y del Credit Default
    Challenge: mismas métricas, mismo split y tiempo de entrenamiento.

    Args:
        modelos: Lista de instancias de modelos del paquete (con el mismo
            DataFrame, target y ``random_state`` para que el split coincida).
        cv: Folds de la validación cruzada estratificada.
        scoring: Métrica de CV a reportar (``'f1'`` para el caso binario).

    Returns:
        DataFrame con una fila por modelo: métricas del test, media y
        desviación de la CV, y segundos de entrenamiento. Ordenado por
        ``balanced_accuracy`` descendente.
    """
    filas = {}
    for m in modelos:
        m.preparar_datos()
        t0 = time.perf_counter()
        m.ajustar()
        segundos = time.perf_counter() - t0
        fila = m.evaluar()
        cvr = m.validacion_cruzada(cv=cv, scoring=(scoring,)).loc[scoring]
        fila.update({f"cv_{scoring}_media": cvr["media"],
                     f"cv_{scoring}_desv": cvr["desv_std"],
                     "segundos": segundos})
        filas[m.nombre] = fila
    return pd.DataFrame(filas).T.sort_values("balanced_accuracy", ascending=False)
