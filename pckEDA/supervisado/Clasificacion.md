# Clasificacion — Clasificación Supervisada (en desarrollo)

## Estado actual

> ⚠️ Esta clase es actualmente un **esqueleto** (placeholder). Los métodos del algoritmo aún no están implementados.

---

## ¿Qué hará cuando esté completa?

La clase `Clasificacion` implementará algoritmos de **aprendizaje supervisado** para predecir a qué **categoría** pertenece una nueva observación, aprendiendo de ejemplos etiquetados.

**Analogía:** Tienes 10,000 reservas de hotel y sabes cuáles se cancelaron y cuáles no. La clasificación aprende de esos ejemplos para predecir si una nueva reserva (de la que sólo conoces los datos, no el resultado) se cancelará o no.

### ¿Qué es "supervisado"?

A diferencia del clustering (no supervisado), aquí el modelo aprende de datos que **ya tienen la respuesta correcta** (la columna objetivo). El modelo busca aprender las reglas que conectan las variables de entrada con esa respuesta.

---

## Creación del objeto (estructura actual)

```python
import pckEDA as mf

clasificacion = mf.Clasificacion(df, target="is_canceled")
```

### Parámetros del constructor

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `df` | `DataFrame` | Tabla de datos con la columna objetivo incluida |
| `target` | `str` | Nombre de la columna que contiene las categorías a predecir |

> **Columna objetivo (`target`):** Debe ser una variable categórica (ej. "Cancelado" / "No cancelado", "Sí" / "No", o una clase de 0 a N).

---

## Algoritmos previstos

| Algoritmo | Cuándo usar |
|-----------|-------------|
| Árbol de decisión | Fácil de interpretar; bueno como punto de partida |
| Random Forest | Alta precisión; robusto a outliers |
| Regresión logística | Cuando necesitas probabilidades; relaciones lineales |
| SVM | Datasets pequeños con separación clara entre clases |

---

## Métricas de evaluación previstas

| Métrica | ¿Qué mide? |
|---------|------------|
| Exactitud (Accuracy) | % de predicciones correctas en total |
| Precisión | De los que predije como positivos, ¿cuántos lo son? |
| Recall | De los que realmente son positivos, ¿cuántos detecté? |
| F1-Score | Equilibrio entre precisión y recall |
| Matriz de confusión | Tabla de errores y aciertos por categoría |

---

## Guía para contribuir

Para implementar esta clase, sigue el patrón de `acp.py` y `hac.py`:

1. Hereda de `Supervisado`
2. Separa features (`X`) y etiquetas (`y`) a partir de `target`
3. Usa `sklearn.model_selection.train_test_split` para dividir train/test
4. Expón resultados como `@property` con name-mangling
5. Implementa al menos: entrenamiento, predicción, métricas y matriz de confusión
6. Registra en los `__init__.py` (ver `pckEDA/README.md`)
