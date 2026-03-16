# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

Planned additions:
- `Clasificacion` — supervised classification algorithms
- `Regresion` — supervised regression algorithms

---

## [0.5.0] — 2026-03-16

### Added
- `KMeans` class (`pckEDA/no_supervisado/kmeans.py`) implementing K-Means clustering
  with `plot_codo`, `plot_silhouette`, `plot_mapa_calor`, `plot_distribucion`,
  `plot_dispersion`; exposes `inercia` and `silhouette` quality metrics
- `TSNE` class (`pckEDA/no_supervisado/tsne.py`) implementing t-SNE dimensionality
  reduction with `plot_proyeccion` (supports external cluster labels for coloring)
  and `plot_perplejidad` for hyperparameter calibration
- `UMAP` class (`pckEDA/no_supervisado/umap.py`) implementing UMAP dimensionality
  reduction with `plot_proyeccion`, `plot_vecinos`, and `transformar()` for projecting
  new data; requires `umap-learn` (available in Anaconda)
- Individual Jupyter notebooks: `KMeans_HotelBookings.ipynb`, `TSNE_HotelBookings.ipynb`,
  `UMAP_HotelBookings.ipynb` in `Semana 9/`
- Comparison notebook `Semana 9/Comparacion_NoSupervisado.ipynb` — side-by-side
  analysis of all five unsupervised algorithms with cross-visualization
  (projections colored by cluster labels)
- Documentation files: `KMeans.md`, `TSNE.md`, `UMAP.md`

### Changed
- `pckEDA/__init__.py` and `pckEDA/no_supervisado/__init__.py` updated to export
  `KMeans`, `TSNE`, `UMAP`
- `pckEDA/README.md` updated with new class index, quick-start examples, and
  updated package tree

---

## [0.4.1] — 2026-03-09

### Changed
- `HAC` constructor now accepts `path` and `num` (same signature as `AnalisisDatosExploratorio`)
  instead of a pre-built DataFrame, making the EDA inheritance chain fully functional
- Fitting logic extracted from `__init__` into a new `ajustar()` method, enabling
  EDA cleaning methods to be called between object creation and model training
- `HAC_HotelBookings.ipynb` updated to use the EDA-first workflow:
  `codificarCategorica()`, `eliminarNulos()`, `analisisNumerico()` → `ajustar()`
- `HAC.md` updated with corrected constructor signature, `ajustar()` documentation,
  and revised code examples

---

## [0.4.0] — 2026-03-09

### Added
- `HAC` class (`pckEDA/no_supervisado/hac.py`) implementing Hierarchical Agglomerative
  Clustering with auto-scaling, cophenetic correlation quality metric, and four
  visualization methods: `plot_dendrograma`, `plot_mapa_calor`, `plot_distribucion`,
  `plot_dispersion`
- `Semana 9/HAC_HotelBookings.ipynb` — Jupyter notebook front-end for HAC
- Per-algorithm documentation files using uppercase class name convention (`ACP.md`,
  `HAC.md`, `Clustering.md`, `EDA.md`, `Clasificacion.md`, `Regresion.md`), each
  covering parameters, properties, methods, examples, and tips for non-statisticians
- `pckEDA/README.md` — new user-facing entry point with class index, quick-start
  examples, and links to all algorithm documentation
- `pckEDA/CONTRIBUTING.md` — renamed from the previous `README.md`; developer guide
  for adding new algorithms to the package

### Changed
- `pckEDA/__init__.py` and `pckEDA/no_supervisado/__init__.py` updated to export `HAC`

---

## [0.3.0] — 2026-03-03

### Added
- `pckEDA` restructured as a proper Python package following PEP 328 (relative imports)
- `ACP` class (`pckEDA/no_supervisado/acp.py`) implementing Principal Component Analysis
  via the `prince` library, with properties for variance explained, variable correlations,
  individual coordinates, contributions, and cos² quality; three visualization methods:
  `plot_plano_principal`, `plot_circulo`, `plot_sobreposicion`
- `pckEDA/README.md` (contributor guide) documenting the two-step process for
  adding new algorithms

### Changed
- Package layout separated into `no_supervisado/` and `supervisado/` submodules,
  each with their own `__init__.py` and base class

---

## [0.2.0] — 2026-03-02

### Added
- `AnalisisDatosExploratorio` class (`pckEDA/eda.py`) — core EDA engine with methods
  for loading (CSV modes 1 and 2), cleaning (`eliminarNulos`, `eliminarDuplicados`,
  `eliminarOutliers`), transformation (`codificarCategorica`, `analisisNumerico`,
  `analisisCompleto`), and visualization (`graficarHeatmap`, `graficarScatter`,
  `graficarBoxplot`, `graficarFrecuencias`, and more)
- `NoSupervisado` and `Supervisado` base classes inheriting from
  `AnalisisDatosExploratorio`, establishing the inheritance chain for all future models
- `Clustering` stub class placeholder for future K-Means implementation
- `Clasificacion` and `Regresion` stub classes for future supervised algorithms
- Jupyter notebooks for Semanas 4, 5, 7 and 8 covering EDA and initial PCA exploration
- Initial PCA notebooks using hotel bookings and student productivity datasets

---

## [0.1.0] — 2026-01-26

### Added
- Initial project setup with `Leccion3.py`
- Object-oriented programming foundations: `Account`, `CurrentAccount`, and
  `SavingAccount` classes demonstrating properties, constructors, and inheritance
- `.gitignore` configuration

---

*Maintainer: Guillermo Jiménez — LEAD University, Minería de Datos I*
