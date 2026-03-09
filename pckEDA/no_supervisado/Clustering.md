# Clustering — K-Means (en desarrollo)

## Estado actual

> ⚠️ Esta clase es actualmente un **esqueleto** (placeholder). Los métodos del algoritmo K-Means aún no están implementados. Consulta `README_HAC.md` para una implementación completa de clustering jerárquico ya disponible.

---

## ¿Qué hará cuando esté completa?

La clase `Clustering` implementará el algoritmo **K-Means**, que divide los datos en `k` grupos buscando minimizar la distancia de cada punto a su centroide (centro del grupo).

**Analogía:** Imagina que quieres repartir 100 clientes en 4 grupos. K-Means elige 4 "puntos centrales" al azar y asigna cada cliente al más cercano. Luego recalcula los centros y repite hasta que los grupos se estabilizan.

### Diferencias clave con HAC

| Característica | K-Means (`Clustering`) | HAC |
|----------------|------------------------|-----|
| Velocidad | Muy rápido en datasets grandes | Más lento en datasets grandes |
| Resultado | Puede variar entre ejecuciones | Siempre el mismo resultado |
| Número de clusters | Debes decidirlo antes | El dendrograma ayuda a elegirlo |
| Forma de clusters | Asume clusters esféricos | Sin restricción de forma |

---

## Creación del objeto (estructura actual)

```python
import pckEDA as mf

clustering = mf.Clustering(df, n_clusters=3)
```

### Parámetros del constructor

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `df` | `DataFrame` | Tabla de datos numéricos |
| `n_clusters` | `int` | Número de grupos a formar |

---

## Guía para contribuir

Para implementar esta clase, sigue el patrón de `acp.py` y `hac.py`:

1. Hereda de `NoSupervisado`
2. Usa `sklearn.cluster.KMeans` para el ajuste
3. Expón resultados como `@property` con name-mangling (`__atributo`)
4. Implementa al menos: dendrograma de inercia (codo), scatter por cluster, distribución de clusters
5. Registra en los `__init__.py` (ver `pckEDA/README.md`)
