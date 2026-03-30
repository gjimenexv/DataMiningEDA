# KNN — Clasificación K-Nearest Neighbors

## Que hace?

**K-Nearest Neighbors (KNN)** clasifica una observación asignándole la clase
mayoritaria entre sus `k` vecinos más cercanos en el espacio de features.
1. Calcula la distancia de la nueva observación a todas las del entrenamiento.
2. Selecciona las `k` más cercanas.
3. Asigna la clase que aparece con mayor frecuencia entre esas `k`.

**Analogía:** Imagina que llegas a una fiesta sin conocer a nadie. Para decidir
a qué grupo unirte, miras a las 5 personas más cercanas a ti — si 3 de ellas
están hablando de deportes, te unes al grupo de deportes.

---

## Cuando usarlo?

- Cuando el dataset es de **tamaño moderado** (KNN guarda todos los datos en memoria).
- Cuando la relación entre features y clase **no es necesariamente lineal**.
- Cuando quieres un modelo **fácil de entender** y sin fase de entrenamiento explícita.
- Como **baseline** antes de probar modelos más complejos.

> **Requisito:** Todos los datos deben ser **numéricos** y estar **escalados**
> (la clase escala automáticamente con StandardScaler o MinMaxScaler).

---

## Flujo de uso

```
mf.KNN(df, target)             # 1. Crear con DataFrame y columna objetivo
knn.preparar_datos()            # 2. Separar train/test y escalar
knn.ajustar()                   # 3. Entrenar el modelo
knn.plot_*()                    # 4. Visualizar resultados
knn.indices_general()           # 5. Evaluar métricas
```

## Creación del modelo

```python
import pckEDA as mf

knn = mf.KNN(df, target='is_canceled', n_neighbors=5)
```

### Parámetros del constructor

| Parámetro | Tipo | Valor por defecto | Descripción sencilla |
|-----------|------|-------------------|----------------------|
| `df` | `DataFrame` | — | Tabla de datos con la columna objetivo incluida |
| `target` | `str` | — | Nombre de la columna con las categorías a predecir |
| `n_neighbors` | `int` | `5` | Cuántos vecinos considerar para la votación |
| `weights` | `str` | `'uniform'` | Cómo pesar los vecinos (ver tabla abajo) |
| `metric` | `str` | `'euclidean'` | Métrica de distancia (ver tabla abajo) |
| `scaler` | `str` | `'standard'` | Tipo de escalado: `'standard'` o `'minmax'` |

#### Función de pesos (`weights`)

| Valor | Descripción | Cuándo usarlo? |
|-------|-------------|----------------|
| `'uniform'` | Todos los vecinos votan igual | **Recomendado** como punto de partida |
| `'distance'` | Vecinos más cercanos tienen más peso | Cuando la cercanía importa mucho |

#### Métrica de distancia (`metric`)

| Valor | Fórmula intuitiva | Cuándo usarla? |
|-------|-------------------|----------------|
| `'euclidean'` | Línea recta entre dos puntos | **Recomendado** — el más común |
| `'manhattan'` | Suma de diferencias absolutas (caminar por cuadras) | Datos con muchas dimensiones |
| `'minkowski'` | Generalización de euclidean y manhattan | Experimentación avanzada |

---

## Métodos principales

### `knn.preparar_datos(test_size=0.2, random_state=42)`
Separa features y target, divide en train/test con estratificación,
y escala los datos automáticamente.

### `knn.aplicar_pca(n_componentes=2)`
Aplica PCA para reducir dimensionalidad antes de entrenar.
Útil para mitigar la **maldición de la dimensionalidad**.

### `knn.ajustar()`
Entrena el modelo KNN. Calcula automáticamente predicciones, probabilidades,
exactitud, matriz de confusión y reporte de clasificación.

### `knn.buscar_mejor_k(k_range, weights_options, cv)`
Usa **GridSearchCV** para encontrar el k óptimo y la mejor estrategia de pesos
mediante validación cruzada.

```python
mejores = knn.buscar_mejor_k(k_range=range(1, 21), cv=5)
print(mejores)  # {'n_neighbors': 7, 'weights': 'distance'}
```

---

## Propiedades

| Propiedad | Qué contiene |
|-----------|--------------|
| `knn.modelo` | Modelo `KNeighborsClassifier` ajustado |
| `knn.predicciones` | Array de predicciones sobre el test set |
| `knn.probabilidades` | Matriz de probabilidades por clase |
| `knn.accuracy` | Exactitud del modelo (0-1) |
| `knn.matriz_confusion` | Matriz de confusión (numpy array) |
| `knn.reporte` | Reporte con precision, recall, f1-score |
| `knn.mejores_params` | Mejores hiperparámetros de GridSearchCV |
| `knn.nombres_clases` | Lista ordenada de las clases del target |
| `knn.X_train` / `knn.X_test` | DataFrames de features (sin escalar) |
| `knn.y_train` / `knn.y_test` | Series de etiquetas |

---

## Métodos de evaluación (adaptados de ModMC.py)

### `knn.indices_general(nombres=None)`
Calcula precisión global, error global y precisión por categoría.

```python
indices = knn.indices_general(nombres=['No cancelado', 'Cancelado'])
print(f"Precisión Global: {indices['Precisión Global']:.4f}")
print(f"Error Global:     {indices['Error Global']:.4f}")
print(indices['Precisión por categoría'])
```

### `knn.indices_binarios()`
Solo para problemas **binarios** (2 clases). Calcula VP, VN, FP, FN y métricas
derivadas: PP, PN, PFP, PFN, AP, AN.

```python
metricas = knn.indices_binarios()
if metricas:
    for clave, valor in metricas.items():
        print(f"{clave}: {valor:.4f}")
```

---

## Métodos de visualización

### Gráficos propios de KNN

#### `knn.plot_matriz_confusion(titulo='...')`
Heatmap de la matriz de confusión con las clases etiquetadas.

#### `knn.plot_accuracy_vs_k(k_range=range(1, 31), titulo='...')`
Curva de exactitud para distintos valores de k. El pico indica el k óptimo.

### Gráficos de poder predictivo (adaptados de ModMC.py)

#### `knn.plot_distribucion_target(ax=None)`
Barra horizontal apilada con la proporción de cada clase en el target.

#### `knn.plot_poder_predictivo_categorica(var, ax=None)`
Barras apiladas mostrando la distribución de una variable categórica
según la clase objetivo. Útil para evaluar qué variables categóricas
tienen **poder predictivo**.

#### `knn.plot_poder_predictivo_numerica(var)`
Curvas KDE de una variable numérica, separadas por clase.
Útil para evaluar si una variable numérica **discrimina** entre clases.

---

## Ejemplo completo

```python
import matplotlib.pyplot as plt
import pckEDA as mf

# 1. Crear y preparar
knn = mf.KNN(df, target='is_canceled', n_neighbors=5)

# 2. Explorar poder predictivo
knn.plot_distribucion_target()
plt.show()

knn.plot_poder_predictivo_categorica('hotel')
plt.show()

knn.plot_poder_predictivo_numerica('lead_time')
plt.show()

# 3. Preparar datos y entrenar
knn.preparar_datos(test_size=0.3, random_state=42)
knn.ajustar()

# 4. Evaluar
print(f"Exactitud: {knn.accuracy:.4f}")
print(knn.reporte)
print(knn.indices_general(nombres=['No cancelado', 'Cancelado']))

# 5. Buscar mejor k
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
plt.sca(axes[0]); knn.plot_accuracy_vs_k()
plt.sca(axes[1]); knn.plot_matriz_confusion()
plt.tight_layout(); plt.show()

mejores = knn.buscar_mejor_k()
print(f"Mejores parámetros: {mejores}")

# 6. Re-entrenar con mejores parámetros
knn.n_neighbors = mejores['n_neighbors']
knn.weights = mejores['weights']
knn.ajustar()
print(f"Nueva exactitud: {knn.accuracy:.4f}")
```

---

## Consejos para no estadísticos

- **Escalar siempre:** KNN mide distancias — si una variable va de 0 a 1000 y otra de 0 a 1, la primera domina. La clase escala automáticamente.
- **k impar para 2 clases:** Evita empates usando un k impar (3, 5, 7...).
- **Usa `plot_accuracy_vs_k()`** antes de decidir — el k donde la curva alcanza su pico es el óptimo.
- **Demasiados features?** Aplica `aplicar_pca()` para reducir dimensiones y mejorar velocidad y rendimiento.
- **Clases desbalanceadas?** Revisa `plot_distribucion_target()` — si una clase domina mucho, considera técnicas de balanceo externas antes de crear el objeto.
- **`'distance'` vs `'uniform'`:** Si las clases tienen fronteras difusas, `'distance'` suele funcionar mejor porque da más peso a los vecinos realmente cercanos.
