# Credit Default ML Challenge

**Competencia de Aprendizaje Automático Supervisado**

- **Modalidad:** Grupal (4 estudiantes por equipo)
- **Herramienta:** Python – Google Colab
- **Dataset:** Default of Credit Card Clients
- **Fuente:** UCI Machine Learning Repository

---

## 1. Contexto del desafío

Una institución financiera desea mejorar su capacidad para identificar clientes con riesgo de incumplimiento en el pago de sus tarjetas de crédito.

Para ello, dispone de información histórica de 30.000 clientes, incluyendo características demográficas, límites de crédito, historial de pagos, estados de cuenta y montos pagados.

La institución ha decidido contratar a cinco equipos de científicos de datos para desarrollar un modelo de aprendizaje automático capaz de predecir si un cliente incumplirá su próximo pago.

> **El desafío:** ¿Qué equipo logrará desarrollar el modelo predictivo con el mejor desempeño para identificar correctamente ambas clases de clientes?

---

## 2. Objetivo general

Desarrollar y optimizar modelos de aprendizaje supervisado para predecir el incumplimiento de pagos de tarjetas de crédito, aplicando análisis exploratorio de datos (EDA), ingeniería de variables, validación cruzada y técnicas de evaluación de modelos.

---

## 3. Etapas del Challenge

### Etapa 1. Análisis exploratorio de datos (EDA)

Cada equipo deberá realizar un análisis exploratorio que permita comprender las características del conjunto de datos.

Se deberá incluir:

- Análisis de la estructura y calidad de los datos.
- Identificación de valores faltantes, duplicados y atípicos.
- Análisis de la distribución de la variable objetivo.
- Estudio de las relaciones entre variables.
- Al menos cinco visualizaciones relevantes, acompañadas de su interpretación.

> **Pregunta clave:** ¿Qué patrones o características parecen estar relacionados con el incumplimiento de pagos?

### Etapa 2. Ingeniería de variables

Cada equipo deberá investigar y aplicar técnicas de preparación y transformación de datos que contribuyan a mejorar el rendimiento de sus modelos.

- Las técnicas seleccionadas deberán estar justificadas.
- Se valorará especialmente la creación de nuevas variables a partir de las características originales del dataset.

### Etapa 3. Construcción de modelos

Cada equipo deberá implementar y comparar al menos tres algoritmos de aprendizaje supervisado.

Los equipos tendrán libertad para seleccionar sus algoritmos, realizar ajustes de hiperparámetros y experimentar con diferentes estrategias de entrenamiento.

**Requisitos:**

- Implementar al menos tres modelos de clasificación.
- Aplicar validación cruzada estratificada de 5 folds.
- Realizar experimentos de optimización de hiperparámetros.
- Comparar los resultados obtenidos.
- Seleccionar y justificar el mejor modelo.

### Etapa 4. Evaluación de los modelos

Cada equipo deberá evaluar el rendimiento de sus modelos utilizando las siguientes métricas:

- Accuracy global.
- Precision por clase.
- Recall por clase.
- F1-score por clase.
- Balanced Accuracy.
- Matriz de confusión.

Los resultados deberán presentarse mediante tablas y visualizaciones comparativas.

> **Importante:** obtener un accuracy global elevado no garantiza un buen modelo. Se deberá analizar especialmente la capacidad de identificar clientes que incumplen sus pagos (clase 1).

### Etapa 5. Competencia final

Cada equipo seleccionará su mejor modelo para participar en la competencia.

**El equipo ganador será aquel que obtenga el mayor F1-Score sobre el conjunto de prueba.**
