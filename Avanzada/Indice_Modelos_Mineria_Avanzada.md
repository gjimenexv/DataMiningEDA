# Índice de Modelos — Minería de Datos Avanzada

**Curso:** BCE0019 Minería de Datos Avanzada para Ciencia de Datos (LEAD University, III-2026)
**Profesor:** Rachit Vargas López
**Cobertura:** Semanas 1 a 6 (actualizado al 28-oct-2026)

---

## Tabla resumen

| Sem. | Modelo / tema | Tipo | Idea en una línea | Material en la carpeta |
|---|---|---|---|---|
| 1 | [KNN + métricas de clasificación](#semana-1--k-nearest-neighbors-knn) | Basado en instancias, no paramétrico | Clasifica según la clase mayoritaria de los *K* vecinos más cercanos | `KNN.pptx`, Tarea 1 (matriz de confusión), caso *Bank of London* |
| 2 | [Árbol de decisión (CART)](#semana-2--árbol-de-decisión-cart) | Basado en reglas | Divide los datos con preguntas que minimizan la impureza (Gini) | `Arbol de decision.pptx`, Tarea KNN vs DT |
| 3 | [Random Forest + tuneo + Bagging](#semana-3--random-forest-bagging-y-tuneo-de-hiperparámetros) | Ensamble paralelo (bagging) | Promedia/vota muchos árboles entrenados en muestras bootstrap | `Random Forest.pptx`, Tarea tuneo (Grid/Random/Bayes) |
| 4 | [AdaBoost + Cross-Validation](#semana-4--adaboost-y-validación-cruzada) | Ensamble secuencial (boosting) | Repondera los casos mal clasificados y combina por voto ponderado | `Adaboosting.pdf`, Tarea comparación 4 algoritmos |
| 5 | [Gradient Boosting y XGBoost](#semana-5--gradient-boosting-y-xgboost) | Ensamble secuencial (boosting por gradiente) | Cada árbol aprende los residuos del anterior | `gradient_boosting_xgboost.pdf`, caso *Vértice Retail* (churn) |
| 6 | [LDA y QDA](#semana-6--análisis-discriminante-lda-y-qda) | Probabilístico / lineal | Proyecta los datos para maximizar separación entre clases | `LDA.pptx` |

### Comparación rápida de modelos

| Modelo | ¿Escalar variables? | Frontera | Riesgo principal | Hiperparámetros clave | Interpretabilidad |
|---|---|---|---|---|---|
| KNN | **Sí, obligatorio** | No lineal | Sobreajuste con K bajo, lento en predicción | `n_neighbors`, `metric`, `weights` | Local (por vecinos) |
| Árbol de decisión | No | Ortogonal / escalonada | Alta varianza (sobreajuste) | `max_depth`, `min_samples_leaf`, `criterion`, `ccp_alpha` | Muy alta |
| Random Forest | No | No lineal | Costo computacional, menos interpretable | `n_estimators`, `max_features`, `max_depth` | Media (importancia de variables) |
| AdaBoost | No | No lineal | Muy sensible a ruido y outliers | `n_estimators`, `learning_rate`, `estimator` | Baja |
| Gradient Boosting / XGBoost | No | No lineal | Puede memorizar con muchos árboles | `learning_rate`, `n_estimators`, `max_depth`, `reg_lambda`, `gamma` | Baja (usar SHAP) |
| LDA | Recomendado | **Lineal** | Supuestos de normalidad e igual covarianza | `solver`, `shrinkage` | Alta |
| QDA | Recomendado | Cuadrática | Más parámetros → más sobreajuste | `reg_param` | Media |

### Bagging vs. Boosting (concepto transversal S3–S5)

| | Bagging (Random Forest) | Boosting (AdaBoost, GBM, XGBoost) |
|---|---|---|
| Entrenamiento | Paralelo, modelos independientes | Secuencial, cada modelo corrige al anterior |
| Reduce principalmente | **Varianza** | **Sesgo** (y algo de varianza) |
| Datos de cada modelo | Muestras bootstrap | Datos reponderados (AdaBoost) o residuos (GBM) |
| Combinación | Voto mayoritario / promedio | Suma ponderada |
| Riesgo | Poco sobreajuste | Sobreajusta con ruido o demasiadas iteraciones |

---

## Conceptos base (Semana 1, aplican a todos los modelos)

- **Clasificación vs. regresión:** clasificación predice una clase discreta; regresión un valor continuo.
- **Definición formal:** dada una base $D = \{t_1,\dots,t_n\}$ y clases $C = \{C_1,\dots,C_m\}$, se busca $f: D \to C$.
- **Partición de datos:** ~70 % entrenamiento, ~15 % validación, ~15 % prueba (en las tareas se usa 80/20 con `stratify=y`).
- **Aprendizaje basado en instancias** (KNN: "dime con quién andas y te diré quién eres", guarda los datos) **vs. basado en modelos** (aprende una función/regla global y la aplica a todos).

### Matriz de confusión y métricas

|  | Predicho positivo | Predicho negativo |
|---|---|---|
| **Real positivo** | TP (verdadero positivo) | FN (falso negativo, error tipo II) |
| **Real negativo** | FP (falso positivo, error tipo I) | TN (verdadero negativo) |

| Métrica | Fórmula | Pregunta que responde |
|---|---|---|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | ¿Qué % de todas las predicciones acertó? |
| Precision | TP/(TP+FP) | De los que marqué positivos, ¿cuántos lo eran? |
| Recall (sensibilidad) | TP/(TP+FN) | De los positivos reales, ¿cuántos capturé? |
| Especificidad | TN/(TN+FP) | De los negativos reales, ¿cuántos identifiqué? |
| F1-Score | 2·(P·R)/(P+R) | Media armónica: equilibra Precision y Recall |

**Lecciones clave de la Tarea 1 (churn de telecom, 1 000 clientes):**
- Modelo 1 (TP 180, TN 620, FP 80, FN 120): Accuracy 80 %, Precision 69,23 %, Recall 60 %, Especificidad 88,57 %, F1 64,29 %.
- Modelo 2, con el umbral cambiado (TP 240, TN 560, FP 140, FN 60): Accuracy 80 %, Precision 63,16 %, Recall 80 %, F1 70,59 %.
- Con costo FP = $10 y FN = $100: el modelo 1 cuesta **$12 800** y el modelo 2 **$7 400**. **El mismo Accuracy puede tener costos muy distintos.**
- Bajar el umbral sube el Recall y baja la Precision (*trade-off*). La métrica a priorizar depende del costo de negocio de cada tipo de error.
- Con clases desbalanceadas, el Accuracy engaña: hay que mirar F1, Recall y la matriz completa.

---

## Semana 1 — K-Nearest Neighbors (KNN)

**Qué es:** algoritmo supervisado, **no paramétrico y basado en instancias** (*lazy learning*). No "entrena": guarda todos los datos y, para un punto nuevo, busca los K más cercanos.

**Algoritmo:**
1. Definir los datos de entrenamiento (con su clase).
2. Elegir K.
3. Elegir la métrica de distancia.
4. Almacenar los datos (no se construye ningún modelo).
5. Para una observación nueva, calcular la distancia a todos los puntos y tomar los K más cercanos.
6. Predecir: **clasificación** → clase mayoritaria, $\hat{y} = \arg\max_c \sum_{i\in N_K(x)} I(y_i = c)$. **Regresión** → promedio, $\hat{y} = \frac{1}{K}\sum_{i\in N_K(x)} y_i$.

**Distancia euclidiana:** $d(x_i,x_j) = \sqrt{\sum_{k=1}^{p}(x_{ik}-x_{jk})^2}$. Otras: Manhattan, Minkowski (con parámetro *p*: 1 = Manhattan, 2 = Euclidiana), Chebyshev, Mahalanobis, coseno.

**Efecto de K (sesgo-varianza):**
- **K pequeño (K=1):** fronteras muy complejas que se ajustan al ruido → **overfitting** (alta varianza). Con K=1 el espacio se divide en un **diagrama de Voronoi**: cada punto nuevo toma la clase de su vecino más cercano.
- **K grande (K=100):** fronteras demasiado suaves → **underfitting** (alto sesgo).
- El K óptimo equilibra ambos extremos (en los ejemplos de clase, K=20 generaliza mejor que K=1).

**Hiperparámetros (`KNeighborsClassifier`):** `n_neighbors` (1–50, típicamente 3, 5, 7…), `metric`, `weights` (`uniform` o `distance`, donde los vecinos cercanos pesan más), `p`, `algorithm` (`auto`, `ball_tree`, `kd_tree`, `brute`), `leaf_size`, `n_jobs`.

| Ventajas | Desventajas |
|---|---|
| No paramétrico, no asume forma de f(x) | Inferencia costosa: O(n·p) por predicción |
| Captura fronteras no lineales | Alto consumo de memoria (guarda todo) |
| Entrenamiento trivial | **Maldición de la dimensionalidad** |
| Flexible en la métrica de distancia | **Sensible a la escala** → hay que estandarizar |
| Interpretabilidad local (vecinos) | Sensible a ruido y outliers; no da una función global explícita |

**Ejercicios:** caso *Bank of London* (discriminación salarial por género, 93 empleados: salario inicial, tiempo, educación, experiencia) y Tarea 1 de métricas.

---

## Semana 2 — Árbol de Decisión (CART)

**Qué es:** modelo que toma decisiones mediante **reglas lógicas** en forma de árbol. Cada **nodo** es una pregunta sobre una variable y cada **hoja** es una predicción. Es el modelo más interpretable visto en el curso.

**Índice de Gini (impureza de un nodo):** $G = 1 - \sum_{i=1}^{n} p_i^2$

| Gini | Interpretación |
|---|---|
| 0 | Nodo completamente puro |
| 0,1–0,3 | Bastante puro |
| 0,4–0,5 | Muy mezclado |
| 0,5 | Máxima impureza (2 clases) |

**Algoritmo CART paso a paso** (ejemplo de clase: 24 transacciones, objetivo `Es_Fraudulenta`):
1. **Nodo raíz:** 16 no fraude / 8 fraude → Gini = 1 − (0,667² + 0,333²) = **0,445**.
2. **Evaluar cada atributo:** se calcula el **Gini ponderado** de cada partición posible, $Gini_{split} = \frac{N_L}{N}Gini(L) + \frac{N_R}{N}Gini(R)$, y se elige el menor. En el ejemplo gana **País** (Gini ponderado 0,125) frente a Historial (0,250), Canal (0,146), Edad (0,222) y Monto (0,267).
3. **Dividir** por País: México y Nigeria quedan puros (Gini 0); Brasil queda mezclado (Gini 0,5).
4. **Repetir recursivamente** en las ramas impuras (Brasil se divide por Historial de compras; la rama "Regular" se divide por Canal).
5. Se para cuando las hojas son puras o se cumple un criterio de parada.

**Umbral para variables numéricas:** se ordenan los valores, los candidatos son los puntos medios entre valores consecutivos, $t_i = \frac{x_i + x_{i+1}}{2}$, y se elige el *t* con menor Gini ponderado (en el ejemplo de Monto, t* = 660).

**Hiperparámetros (`DecisionTreeClassifier`):** `criterion` (gini, entropy, log_loss), `splitter`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `max_leaf_nodes`, `ccp_alpha` (poda por costo-complejidad), `class_weight` (`balanced` para clases desbalanceadas), `random_state`.

| Ventajas | Desventajas |
|---|---|
| Muy interpretable | **Alta varianza** (sobreajusta si es profundo) |
| Maneja variables numéricas y categóricas | Inestable: pequeños cambios en los datos dan otro árbol |
| No requiere escalar ni normalizar; robusto a outliers | Menor precisión que los ensambles |
| Captura no linealidades | Sesgo hacia variables con muchos niveles |
| Selecciona variables automáticamente | Árboles grandes se vuelven complejos |

**Tarea KNN vs DT** (Breast Cancer y Wine, split 80/20 estratificado):

| Dataset | Modelo | Accuracy | F1 | CV 5-fold |
|---|---|---|---|---|
| Breast Cancer | Decision Tree | 0,912 | 0,929 | 0,910 |
| Breast Cancer | KNN (k=5, escalado) | **0,956** | **0,966** | **0,963** |
| Wine | Decision Tree | 0,944 | 0,945 | 0,893 |
| Wine | KNN (k=5, escalado) | **0,972** | **0,972** | **0,972** |

Conclusión: KNN ganó en ambos (variables numéricas continuas bien agrupadas tras escalar); el árbol sin podar sobreajusta. Usar `StandardScaler` **dentro de un Pipeline** para evitar fuga de información.

---

## Semana 3 — Random Forest, Bagging y Tuneo de Hiperparámetros

**Motivación:** un solo árbol tiene **alta varianza**: si cambia el conjunto de entrenamiento ($D \to D'$), el árbol $T(D)$ puede ser completamente distinto de $T(D')$. Se busca reducir la varianza sin aumentar mucho el sesgo.

**Ensemble learning:** combinar varios clasificadores diversos da una predicción más robusta, **siempre que sus errores no estén totalmente correlacionados**.

**Random Forest** (Breiman, 2001):
- Se entrenan *B* árboles $T_1(x),\dots,T_B(x)$, cada uno sobre una **muestra bootstrap** (muestreo con reemplazo) y usando un **subconjunto aleatorio de variables en cada división** (`max_features`).
- Clasificación: $\hat{y} = \text{moda}(T_1(x),\dots,T_B(x))$, es decir, votación mayoritaria. Regresión: $\hat{f}_{RF}(x) = \frac{1}{B}\sum_{b=1}^{B} T_b(x)$.

**Por qué reduce la varianza:** si los árboles fueran independientes, $Var(\hat{f}) = \sigma^2 / B$. Con correlación ρ entre árboles: $Var(\hat{f}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$. Por eso se decorrelacionan los árboles con `max_features`.

**Qué pasa con el sesgo:** RF usa árboles profundos (sesgo bajo), así que $Bias_{RF} \approx Bias_{árbol}$. **El sesgo prácticamente no cambia; lo que baja es la varianza.**

**Bootstrap:** en promedio cada muestra contiene ~**63,2 %** de los datos originales ($1-(1-1/n)^n \to 1-e^{-1}$). El ~37 % restante (*out-of-bag*, OOB) sirve para estimar el error sin un set de validación (`oob_score=True`).

**Hiperparámetros (`RandomForestClassifier`):** `n_estimators` (más árboles → menos varianza y más costo; 100–500), `max_depth`, `max_features` (`sqrt` para clasificación), `min_samples_split`, `min_samples_leaf`, `bootstrap`, `oob_score`, `class_weight`.

| Ventajas | Desventajas |
|---|---|
| Disminuye la varianza y el error de generalización | Mayor costo computacional, O(B·N·log N·m) |
| Decorrelación entre árboles | Más memoria |
| Consistencia estadística (converge con B → ∞) | Menos interpretable que un árbol |
| Robusto a ruido y outliers | Rendimientos decrecientes al subir B |
| | Si los árboles están muy correlacionados (ρ ≈ 1), no mejora |

### Tuneo de hiperparámetros

- **Parámetros** se aprenden de los datos (umbrales, coeficientes); **hiperparámetros** se fijan antes de entrenar (K, `max_depth`, `n_estimators`, `learning_rate`).
- Cada combinación se evalúa con **validación cruzada sobre el train**; el test se reserva para la evaluación final.

| Técnica | Cómo elige combinaciones | Costo | ¿Usa historial? |
|---|---|---|---|
| **Grid Search** | Todas las combinaciones de una malla | Alto (crece exponencialmente) | No |
| **Random Search** | Muestreo aleatorio de `n_iter` combinaciones | Controlable | No |
| **Bayesian Search** | Modelo sustituto + función de adquisición | Controlable y eficiente | **Sí** |

**Resultados de la tarea** (Decision Tree en Breast Cancer): árbol default F1 test 0,938 · Grid (672 combinaciones, ~12 s) F1 CV 0,954 / test 0,938 · Random (60 iteraciones, ~1 s) test **0,967** · Bayesian (40 iteraciones, ~33 s) test **0,967** · Bagging de 100 árboles F1 CV **0,961**, AUC 0,994. **Random Search dio el mejor resultado por costo**, y Bagging fue lo que más estabilizó el árbol.

---

## Semana 4 — AdaBoost y Validación Cruzada

**Origen:** Kearns y Valiant (1988–89) preguntan si se puede crear un clasificador robusto a partir de **clasificadores débiles**. Schapire (1990) lo demuestra. Freund y Schapire (1996–97) proponen **AdaBoost.M1** (Premio Gödel). Breiman lo llamó "el mejor clasificador estándar del mundo".

**Idea:** entrenar clasificadores en secuencia y **aumentar el peso de las observaciones mal clasificadas** para que el siguiente les dé prioridad.

**Algoritmo AdaBoost.M1:**
1. Inicializar pesos $w_i = 1/N$.
2. Para m = 1 hasta M:
   - Ajustar el clasificador $G_m(x)$ usando los pesos $w_i$.
   - Calcular el error ponderado: $err_m = \frac{\sum w_i\, I(y_i \neq G_m(x_i))}{\sum w_i}$.
   - Calcular el peso del clasificador: $\alpha_m = \log\left(\frac{1-err_m}{err_m}\right)$.
   - Actualizar: $w_i \leftarrow w_i \cdot \exp[\alpha_m \cdot I(y_i \neq G_m(x_i))]$.
3. Predicción final por **voto mayoritario ponderado**: $G(x) = \text{sign}\left[\sum_{m=1}^{M} \alpha_m G_m(x)\right]$.

**Dinámica de pesos (ejemplo de clase):** 10 observaciones empiezan con peso 0,1. Tras la ronda 1, las 3 mal clasificadas suben a 0,311 y las bien clasificadas bajan a 0,01.

**Métodos de consenso vs. potenciación:** consenso (bagging) = *m* muestras bootstrap y gana el más votado; si todos son árboles, es un Random Forest. Potenciación (boosting) = repondera los errores.

**sklearn:** `AdaBoostClassifier(estimator=None, n_estimators=50, learning_rate=1.0, random_state=None)`. Por defecto usa *stumps* (árboles de profundidad 1).

| Ventajas | Desventajas |
|---|---|
| Mejora clasificadores débiles | **Extremadamente sensible a ruido y outliers** (les asigna mucho peso) |
| Menos propenso a sobreajuste en datos limpios | Más lento que XGBoost |
| Combinable con cualquier algoritmo base | Requiere un dataset limpio |
| Versátil (reconocimiento de personas y emociones, riesgos) | Limitación de escala |

### Validación cruzada (tarea S4, Parte III)
- **k-Fold CV:** divide los datos en *k* partes; entrena con k−1 y valida con la restante, *k* veces. Reporta **media ± desviación estándar**, que es más confiable que un único split.
- **Stratified k-Fold:** mantiene la proporción de clases en cada fold (clave con desbalance).
- **Feature engineering (Parte II):** *normalización* (Min-Max a [0,1]) vs. *estandarización* (z-score, media 0 y desviación 1); selección de variables; codificación de categóricas. El escalamiento es crítico para KNN porque usa distancias.
- **Tarea:** comparar KNN, DT, RF y AdaBoost con el mismo split 80/20 (Accuracy, Precision, Recall, F1, matriz de confusión y **tiempo de entrenamiento**) y luego con Stratified 5-Fold.

---

## Semana 5 — Gradient Boosting y XGBoost

**Idea:** "como un golfista": cada golpe (árbol) no empieza de cero, sino que apunta a lo que falta. Es una predicción inicial más una cadena de pequeñas correcciones.

**Algoritmo Gradient Boosting (clasificación):**
1. **Paso 0:** predicción inicial = proporción de positivos (p. ej. 50 % = 0 en log-odds).
2. **Repetir:**
   a. Calcular el **residuo** = real − predicho.
   b. Entrenar un **árbol pequeño que predice los residuos** (no la clase).
   c. Sumar su corrección multiplicada por la **tasa de aprendizaje** (*learning rate*).
3. Convertir los puntos totales (**log-odds**) a probabilidad con la **sigmoide**.

- Los árboles suman en escala de **log-odds** (puntos), no en porcentajes, para no salirse de [0, 1]. Cerca de 50 %, 1 punto ≈ 25 % de probabilidad.
- **¿Por qué "gradiente"?** El residuo es la dirección cuesta abajo de la función de pérdida: es bajar la montaña del error, un árbol por paso.
- **Ejemplo de clase:** un cliente con 4 quejas pasa de 50 % → 73 % → 84 % → 87 % de probabilidad de irse tras 3 árboles (0 + 1,00 + 0,68 + 0,20 = 1,88 puntos).
- **Cuidado:** demasiados árboles memorizan los casos raros. Solución: learning rate pequeño (0,01–0,1) y **early stopping**.

**XGBoost** (Chen y Guestrin, 2016) es el mismo juego "mejor jugado":
1. **Es prudente (λ, `reg_lambda`):** desconfía de hojas con pocos datos. Con λ = 1, una hoja de 1 cliente que pediría +2 se reduce a +0,40; con 100 clientes, +1,92.
2. **Elige mejores cortes (ganancia) y poda (γ, `gamma`):** un corte solo se mantiene si su ganancia es mayor que γ. Más γ → árboles más simples.
3. **Es rápido y práctico:** maneja **valores faltantes** solo (aprende la dirección por defecto), usa histogramas (~256 cubetas), paralelismo, submuestreo y GPU.

| Hiperparámetro | Significado | Valor típico |
|---|---|---|
| `n_estimators` | Número de árboles | Muchos + early stopping |
| `learning_rate` | Tamaño de cada corrección | 0,01–0,1 |
| `max_depth` | Preguntas por árbol | 3–6 |
| `reg_lambda` (λ) | Desconfianza en hojas pequeñas | 1–10 |
| `gamma` (γ) | Costo de cada corte extra | 0–5 |
| `subsample` | Fracción de datos por árbol | 0,7–1 |
| `scale_pos_weight` | Peso de la clase minoritaria | n.º negativos / n.º positivos |

**Diagnóstico:** si sobreajusta → menos profundidad, más λ y γ, tasa más pequeña. Si subajusta → más profundidad, más árboles, menos regularización.

**Clases desbalanceadas:** un modelo que dice "nadie se va" puede tener 90 % de Accuracy y no servir. Usar **AUC**, **AUC-PR** (pocos positivos) y **log-loss**.

| | Gradient Boosting | XGBoost |
|---|---|---|
| Hojas con pocos datos | Les cree | Desconfía (λ) |
| Tamaño del árbol | Profundidad fija | Poda cortes que no valen la pena (γ) |
| Datos faltantes | Hay que imputar | Los maneja solo |
| Velocidad | Lento con muchos datos | Muy rápido |

**Caso Vértice Retail (churn e-commerce, 5 630 clientes, 20 variables):**
- *Unit economics:* la 1.ª compra pierde −$5,25 (CAC $28); las recompras ganan +$27,40. El 62 % de los clientes nunca recompra.
- Retos: (1) *feature importance* con Regresión Logística, RF y XGBoost, más la interacción Complain × SatisfactionScore × Tenure; (2) **threshold tuning económico**: incentivo de $10 con 30 % de efectividad, simulando umbrales de 0,01 a 0,99 para maximizar la ganancia neta frente a p = 0,50; (3) matriz 2×2 riesgo × valor (incluye la estrategia de "no intervenir"); (4) ética y explicabilidad (SHAP, LIME).
- Matriz financiera: VP = 0,3·(LTV − 10) − 0,7·10 · FP = −$10 · FN = −$32,40 (LTV perdido) · VN = $0.

---

## Semana 6 — Análisis Discriminante (LDA y QDA)

**Qué es LDA:** método supervisado de clasificación **y** de reducción de dimensionalidad. Busca la proyección lineal que **maximiza la separación entre las medias de las clases** y **minimiza la dispersión dentro de cada clase**.

**Supuestos:**
1. Linealidad (frontera = hiperplano).
2. **Normalidad multivariante** en cada clase: $X \mid y=k \sim N(\mu_k, \Sigma)$.
3. **Igualdad de matrices de covarianza** (homocedasticidad): $\Sigma_1 = \dots = \Sigma_K$. Es el supuesto más importante.
4. Independencia de las observaciones.
5. Variables continuas, sin colinealidad perfecta.

**Matemática:**
- Medias: $\mu_k = \frac{1}{n_k}\sum_{i:y_i=k} x_i$ y media global $\mu = \sum \pi_k \mu_k$.
- **Dispersión dentro de clases:** $S_W = \sum_k \sum_{i:y_i=k}(x_i-\mu_k)(x_i-\mu_k)^T$.
- **Dispersión entre clases:** $S_B = \sum_k n_k(\mu_k-\mu)(\mu_k-\mu)^T$.
- **Criterio de Fisher:** $\max_w J(w) = \dfrac{w^T S_B\, w}{w^T S_W\, w}$ (S_B grande, S_W pequeña).
- **Solución:** $S_W^{-1} S_B\, w = \lambda w$; $w^*$ es el autovector del mayor autovalor.
- Como mucho hay **K − 1 direcciones** discriminantes, porque rango(S_B) ≤ K − 1.
- **Clasificación:** proyectar $z = w^{*T}x$ y asignar a la clase cuyo centroide proyectado está más cerca. Con covarianzas iguales y normalidad, **equivale a la regla óptima de Bayes**.
- No es invariante a la escala, así que se recomienda estandarizar.

**Procedimiento:** calcular μ_k y μ → calcular S_W y S_B → resolver S_W⁻¹S_B w = λw → proyectar → clasificar por el centroide más cercano.

| Ventajas | Desventajas |
|---|---|
| Simple e interpretable | Supuestos fuertes (normalidad, igual covarianza) |
| Muy eficiente computacionalmente | Solo fronteras lineales (alto sesgo si son curvas) |
| Funciona bien con pocos datos (menos parámetros) | Sensible a outliers (afectan medias y covarianza) |
| Reduce dimensionalidad (hasta K−1) | Malo con covarianzas muy diferentes |
| Equivale a la regla de Bayes bajo sus supuestos | No captura relaciones no lineales |
| Estable | Pensado para variables continuas |

**Cuándo usar LDA:** clases aproximadamente normales con covarianzas similares, fronteras lineales, pocos datos, necesidad de interpretabilidad.
**Cuándo evitarlo:** covarianzas muy distintas, fronteras no lineales, muchos outliers, variables mayormente categóricas.

### QDA (Quadratic Discriminant Analysis)
Se usa cuando **no se cumple la igualdad de covarianzas**: cada clase tiene su propia $\Sigma_k$.

| | LDA | QDA |
|---|---|---|
| Covarianza | Igual entre clases | Diferente por clase |
| Frontera | Lineal | Cuadrática |
| Parámetros a estimar | Menos | Más |
| Riesgo de sobreajuste | Menor | Mayor |

---

## Guía rápida: ¿qué modelo usar?

| Situación | Modelo sugerido |
|---|---|
| Necesito explicar las reglas al negocio | Árbol de decisión (podado) o LDA |
| Pocos datos, clases ~normales, frontera lineal | LDA |
| Covarianzas distintas entre clases | QDA |
| Variables numéricas escaladas, dataset pequeño/mediano | KNN |
| Quiero robustez y estabilidad con poco tuneo | Random Forest |
| Máxima precisión en datos tabulares | XGBoost / Gradient Boosting (con early stopping) |
| Datos limpios, quiero potenciar un modelo débil | AdaBoost |
| Clases desbalanceadas | `class_weight='balanced'`, `scale_pos_weight`, métricas F1/Recall/AUC-PR, ajuste de umbral |

## Flujo de trabajo común en las tareas

1. Cargar datos → separar X / y.
2. `train_test_split(test_size=0.2, stratify=y, random_state=42)`.
3. Preprocesar **dentro de un Pipeline** (escalar para KNN/LDA, codificar categóricas).
4. Entrenar y evaluar: Accuracy, Precision, Recall, F1, matriz de confusión (+ AUC y tiempo).
5. Validar con **Stratified k-Fold** (media ± desviación estándar).
6. Tunear con Grid, Random o Bayesian Search (CV sobre el train).
7. Interpretar con contexto de negocio: costo de FP vs. FN y ajuste del umbral.

---
*Próximos temas según el programa: Regresión lineal/logística, SVM, LightGBM, regularización (Ridge/Lasso), SHAP, Deep Learning (MLP, CNN, RNN/LSTM) y Series de Tiempo (ARIMA, Holt-Winters, Prophet).*
