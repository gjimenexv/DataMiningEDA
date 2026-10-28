# Changelog — pckModelosAvanzados

All notable changes to this package will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this package uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each new course week that introduces a model adds a `minor` release.

---

## [Unreleased]

Planned additions (per the course syllabus, see
`Avanzada/Indice_Modelos_Mineria_Avanzada.md`):
- Regresión lineal / logística, SVM, LightGBM
- Regularización (Ridge / Lasso) and SHAP explanations
- Deep Learning (MLP, CNN, RNN/LSTM) and time series (ARIMA, Holt-Winters, Prophet)

---

## [0.1.0] — 2026-10-28

Initial release covering Semanas 1 to 6.

### Added
- `ModeloBase` (`base.py`) — shared workflow for every model: stratified
  train/test split, scaling inside a `Pipeline` (no data leakage), `ajustar()`,
  `predecir()`, `evaluar()` (accuracy, balanced accuracy, per-class
  precision/recall/F1, AUC), `indices_binarios()`, stratified k-fold
  `validacion_cruzada()`, `buscar_hiperparametros()` (grid / random / bayes),
  cost-based `optimizar_umbral()`, and confusion-matrix, ROC and feature
  importance plots
- `comparar_modelos()` (`comparar.py`) — side-by-side metrics, CV and training
  time for several models on the same split
- **Semana 1** — `KNN` (`knn/`) with `buscar_mejor_k`, `plot_curva_k`,
  `plot_fronteras_decision`
- **Semana 2** — `ArbolDecision` (`arbol_decision/`) with `gini`,
  `gini_ponderado`, `mejor_umbral`, `reglas_texto`, `ruta_poda`, `plot_arbol`
- **Semana 3** — `Bagging` (`bagging/`) and `RandomForest` (`random_forest/`)
  with out-of-bag score; hyperparameter tuning available via the base class
- **Semana 4** — `AdaBoost` (`adaboost/`) with `pesos_clasificadores` and
  `plot_curva_iteraciones`; cross-validation available via the base class
- **Semana 5** — `GradientBoosting` (`gradient_boosting/`) and `XGBoost`
  (`xgb/`, optional dependency, early stopping, `scale_pos_weight='auto'`)
- **Semana 6** — `LDA` (`lda/`, with projection and covariance diagnostics) and
  `QDA` (`qda/`)
- Per-model usage guides (`KNN.md`, `ArbolDecision.md`, `Bagging.md`,
  `RandomForest.md`, `AdaBoost.md`, `GradientBoosting.md`, `XGBoost.md`,
  `LDA.md`, `QDA.md`)
- `README.md` and `CONTRIBUTING.md` for the package
