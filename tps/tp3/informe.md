# Trabajo Práctico 3: Machine Learning - Clasificación de Tweets sobre Desastres

* **Competencia de Kaggle:** [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)
* **Objetivo:** Determinar si un tweet está relacionado con un desastre real (target=1) o no (target=0).
* **Fecha de entrega:** 20/11/2025
* **Autor:** Máximo Gismondi
* **Repositorio:** [cdd-martinelli](https://github.com/maximogismondi/cdd-martinelli)

Datos del curso:

* **Corrector:** Martín Stefanelli
* **Materia:** Ciencia de Datos
* **Cátedra:** Martinelli
* **Cuatrimestre:** 2025C2

Recomiendo ver el informe en el repositorio para una mejor visualización de tablas e imágenes y navegación entre notebooks.

## Aclaración

Dado la NO obligatoriedad del informe en el TP3, este informe está generado con inteligencia artificial a apartir de los notebooks desarrollados. Mi idea era tener un resumen estructurado de todo el trabajo realizado, con enlaces a los notebooks correspondientes para profundizar en cada sección. La idea era explicar las decisiones tomadas, resultados obtenidos y análisis de los modelos. Además profundizar un poco más en los conceptos detrás de las features y modelos usados.

De todas formas está revisado y corregido manualmente por mi cuenta para asegurarme que tranmite lo que fui haciendo en el TP.

---

## Tabla de Contenidos

1. [Exploración de Datos](#1-exploración-de-datos)
2. [Feature Engineering](#2-feature-engineering)
3. [Modelos Baseline](#3-modelos-baseline)
4. [Modelos Avanzados](#4-modelos-avanzados)
5. [Consigna Extra: Análisis ROC, Feature Importance y Matriz de Confusión](#5-consigna-extra-análisis-roc-feature-importance-y-matriz-de-confusión-1-punto)
6. [Modelo Extra: 4 Features](#6-modelo-extra-4-features-1-punto)
7. [Comparación de Modelos](#7-comparación-de-modelos)
8. [Respuestas a Preguntas de la Consigna](#8-respuestas-a-preguntas-de-la-consigna)
9. [Conclusiones y Lecciones Aprendidas](#9-conclusiones-y-lecciones-aprendidas)

---

## 1. Exploración de Datos

**Notebook:** [`1_data_exploration/exploration.ipynb`](./1_data_exploration/exploration.ipynb)

### 1.1 Descripción del Dataset

El dataset contiene tweets con las siguientes columnas:

- **id**: Identificador único del tweet
- **text**: Contenido del tweet
- **keyword**: Palabra clave asociada (puede ser nula)
- **location**: Ubicación desde donde se envió (puede ser nula)
- **target**: Etiqueta binaria (1=desastre real, 0=no desastre) - solo en train

**Características generales:**
- **Train:** 7,613 tweets
- **Test:** 3,263 tweets
- **Balance:** ~43% de tweets relacionados con desastres (relativamente balanceado)
- **Keywords:** 221 keywords únicas, presente en ~99% de tweets
- **Locations:** 3,341 ubicaciones únicas, presente en ~33% de tweets

### 1.2 Análisis de Features del Texto

Se extrajeron métricas básicas del texto para identificar patrones:

| Feature | Descripción | Insight Clave |
|---------|-------------|---------------|
| `text_length` | Número de caracteres | Tweets de desastres tienden a ser más largos (explican más la situación) |
| `word_count` | Número de palabras | Correlación positiva con desastres |
| `hashtag_count` | Número de hashtags (#) | Máximo de 14 hashtags por tweet |
| `mention_count` | Número de menciones (@) | Desastres tienden a tener menos menciones (menos interacción social) |
| `url_count` | Número de URLs | **Correlación fuerte**: desastres comparten más links a noticias |
| `uppercase_percentage` | Ratio de mayúsculas | Tweets con alta ratio suelen ser spam/no desastre |
| `punctuation_percentage` | Ratio de puntuación | Outliers con mucha puntuación son no-desastres |

**Matriz de Correlación:** Se observó que `url_count` y `text_length` tienen la correlación más alta con el target.

### 1.3 Análisis de Keywords

**Hallazgos clave:**
- Keywords más frecuentes: "fatalities", "derailment", "armageddon", "damage", "deluge"
- **Keywords ambiguas:** "facilities", "damage" (50/50 entre desastre y no-desastre)
- **Keywords claramente NO desastre:** "deluge", "armageddon", "body bags" (>90% no-desastre)
- **Keywords de desastre:** "evacuate", "wildfire", "bombing", "debris" (>80% desastre)

**Insight importante:** Las keywords son altamente predictivas, pero debemos evitar overfitting. Se debe usar junto con análisis del texto completo.

### 1.4 Análisis de Ubicaciones

- Solo 4 ubicaciones tienen ≥30 tweets
- Dominan ubicaciones de Estados Unidos (USA, New York, United States)
- **Conclusión:** Location tiene poco valor predictivo debido a alta fragmentación

### 1.5 Lematización y Análisis de Palabras

**Proceso aplicado:**
1. Conversión a minúsculas
2. Remoción de menciones (@usuario)
3. Remoción de URLs
4. Reemplazo de emojis por espacios
5. Remoción de símbolos especiales
6. Lematización con Spacy (`en_core_web_sm`)
7. Filtrado de stop words

**Palabras más asociadas a desastres:**
- "fire", "kill", "news", "bombing", "wildfire", "mh37" (vuelo desaparecido)

**Palabras más asociadas a NO desastres:**
- "love", "lol", "reddit", "fuck", "like", "new"

**Visualización:** Se creó un Word Cloud con forma del logo de Twitter, donde el color indica la asociación con desastres (rojo) o no (azul), y el tamaño representa la frecuencia.

---

## 2. Feature Engineering

Basándonos en la exploración, se diseñó un pipeline de features:

### 2.1 Features Numéricas

Extraídas directamente del texto raw:
- `text_length`, `word_count`, `char_count`
- `hashtag_count`, `mention_count`, `url_count`
- `punct_count`, `uppercase_percentage`, `punctuation_percentage`
- `stopwords_count`

### 2.2 Features de Texto: TF-IDF + SVD

**Justificación:** 
- **TF-IDF** captura la importancia de palabras en el corpus
- **SVD (Truncated SVD)** reduce dimensionalidad manteniendo información relevante
- Se usan **n-grams** (uni, bi, tri-gramas) para capturar contexto

**Configuración:**
- **TF-IDF Words:** max_features=5000, ngram_range=(1,2), min_df=2
- **TF-IDF Chars:** analyzer='char', ngram_range=(3,5), max_features=3000
- **SVD:** n_components=300

### 2.3 Mean Encoding para Keywords

**¿Por qué Mean Encoding?**
- Captura la probabilidad de desastre dada una keyword
- Más efectivo que One-Hot para categorías con alta cardinalidad
- **Evita data leakage:** Se calcula SOLO sobre el train set antes del split


### 2.4 One-Hot Encoding (Top Keywords)

Se aplica One-Hot Encoding sobre los **top 20 keywords** más frecuentes para capturar su presencia de forma binaria.

**Justificación:** Complementa Mean Encoding al dar información categórica explícita.

### 2.5 Lematización del Texto

Se usa **Spacy** para lematizar el texto, reduciendo palabras a su forma raíz (e.g., "running" → "run").

**Beneficio:** Agrupa variaciones de palabras bajo un mismo lema, mejorando generalización.

---

## 3. Modelos Baseline

### 3.1 Modelo Random (Baseline Mínimo)

**Notebook:** [`2_baseline/random.ipynb`](./2_baseline/random.ipynb)

**Estrategia:** `DummyClassifier` con estrategia "stratified" (predice aleatoriamente según distribución de clases).

**Resultado:**
- **F1-Score:** 0.4565
- **ROC AUC:** 0.52 (línea diagonal, sin capacidad predictiva real)

**Justificación:** Establece el piso mínimo. Cualquier modelo debe superar este baseline.

### 3.2 Logistic Regression

**Notebook:** [`2_baseline/logistic_regression.ipynb`](./2_baseline/logistic_regression.ipynb)

**Arquitectura:**
- Features numéricas (7) + TF-IDF (100 words) + One-Hot Keywords (100 categorías)
- Scaling con `StandardScaler`
- Lematización del texto

**Hiperparámetros:**
- Grid Search sobre `C=[0.1, 1, 10]`, `penalty=['l1', 'l2']`, `solver=['lbfgs', 'liblinear', 'saga']`
- `class_weight='balanced'` para compensar desbalance

**Resultado:**
- **Mejores params:** `C=1`, `penalty='l2'`, `solver='saga'`
- **F1-Score (CV 3-fold):** 0.6970
- **F1-Score (Validation):** 0.7020
- **Accuracy (Validation):** 0.7413
- **Precision:** 0.6946
- **Recall:** 0.7095
- **Score Kaggle:** 0.72632

**Análisis de Coeficientes:**

**Top 10 Features que empujan hacia Disaster (coeficientes positivos):**
1. `text_hiroshima` (3.7)
2. `text_suicide` (2.8)
3. `kw_debris` (2.6)
4. `kw_oil spill` (2.5)
5. `text_wildfire` (2.4)
6. `text_kill` (2.4)
7. `text_fire` (2.3)
8. `text_train` (2.3)
9. `kw_typhoon` (2.1)
10. `text_mass` (2.0)

**Top 10 Features que empujan hacia No-Disaster (coeficientes negativos):**
1. `text_bag` (-1.8)
2. `kw_stretcher` (-1.7)
3. `text_love` (-1.7)
4. `kw_wrecked` (-1.6)
5. `kw_fire` (-1.5)
6. `kw_body bags` (-1.5)
7. `kw_fear` (-1.5)
8. `kw_armageddon` (-1.4)
9. `text_let` (-1.3)
10. `kw_blazing` (-1.3)

**Threshold Optimizado:** 0.5538 (usando Youden's J)
- Con threshold optimizado: **Accuracy = 0.7498**

---

## 4. Modelos Avanzados

Todos los modelos avanzados comparten el **mismo pipeline de features** para comparabilidad:
- TF-IDF + SVD (300 dims)
- Features numéricas escaladas
- Mean Encoding + One-Hot Encoding (top 20 keywords)

### 4.1 Random Forest

**Notebook:** [`3_models/random_forest.ipynb`](./3_models/random_forest.ipynb)

**¿Por qué Random Forest?**
- **Ensemble method:** Combina múltiples árboles de decisión para reducir overfitting
- **Captura no-linealidades:** Puede identificar interacciones complejas entre features
- **Robusto a outliers:** No requiere normalización estricta
- **Interpretable:** Provee feature importance

**Hiperparámetros (RandomizedSearchCV):**
- `n_estimators`: [200, 300, 500]
- `max_depth`: [10, 20, 30, None]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]
- `max_features`: ['sqrt', 'log2']

**Resultado:**
- **Mejores params:** `n_estimators=200`, `max_depth=None`, `min_samples_split=10`, `min_samples_leaf=2`, `max_features='sqrt'`
- **F1-Score (CV 5-fold):** 0.7248
- **F1-Score (Validation):** 0.7564
- **Threshold optimizado:** 0.370
- **ROC AUC:** 0.85
- **Score Kaggle:** 0.76248

**Feature Importance:**

**Top 15 features más importantes:**
1. **Text SVD (Aggregated):** ~84% (dominante, captura la semántica del texto)
2. `keyword_mean_enc`: ~8% (Mean Encoding de keywords)
3. `url_count`: ~1.5%
4. `char_count`: ~1%
5. `punct_count`: ~0.8%
6. `word_count`: ~0.7%
7. `stopwords_count`: ~0.6%
8. `mention_count`: ~0.5%
9. `hashtag_count`: ~0.4%
10. `keyword_top_other`: ~0.3%
11. `has_location_0`: ~0.2%
12. `has_location_1`: ~0.2%
13. `keyword_top_evacuate`: ~0.1%
14. `keyword_top_fatalities`: ~0.1%
15. `keyword_top_famine`: ~0.1%

### 4.2 XGBoost

**Notebook:** [`3_models/xgboost.ipynb`](./3_models/xgboost.ipynb)

**¿Por qué XGBoost?**
- **Gradient Boosting:** Construye árboles secuencialmente, corrigiendo errores previos
- **Regularización incorporada:** L1/L2 para evitar overfitting
- **Manejo eficiente de datos sparse:** Ideal para TF-IDF
- **State-of-the-art performance:** Ganador de múltiples competencias de Kaggle

**Hiperparámetros (RandomizedSearchCV):**
- `max_depth`: [6, 8, 10, 12]
- `learning_rate`: [0.01, 0.05, 0.1]
- `n_estimators`: [300, 500, 800]
- `subsample`: [0.7, 0.8, 0.9]
- `colsample_bytree`: [0.5, 0.7]
- `min_child_weight`: [1, 3, 5]
- `gamma`: [0, 0.1, 0.2]

**Resultado:**
- **Mejores params:** `max_depth=6`, `learning_rate=0.05`, `n_estimators=800`, `subsample=0.9`, `colsample_bytree=0.7`, `min_child_weight=5`, `gamma=0`
- **F1-Score (CV 5-fold):** 0.7525
- **F1-Score (Validation):** 0.7698
- **Threshold optimizado:** 0.420
- **ROC AUC:** 0.87
- **Score Kaggle:** 0.77719

**Feature Importance:**

**Top 15 features más importantes:**
1. **Text SVD (Aggregated):** ~90% (domina aún más que en RF)
2. `keyword_mean_enc`: ~4%
3. `url_count`: ~3%
4. `word_count`: ~0.5%
5. `char_count`: ~0.4%
6. `punct_count`: ~0.3%
7. `stopwords_count`: ~0.2%
8. `hashtag_count`: ~0.2%
9. `mention_count`: ~0.2%
10. `has_location_1`: ~0.1%
11. `has_location_0`: ~0.1%
12. `keyword_top_armageddon`: <0.1%
13. `keyword_top_body%20bags`: <0.1%
14. `keyword_top_collided`: <0.1%
15. `keyword_top_damage`: <0.1%

**Análisis:** XGBoost supera a Random Forest (+1.3 puntos de F1) gracias a su enfoque secuencial de corrección de errores. El modelo concentra aún más importancia en Text SVD (90% vs 84% de RF).

### 4.3 Red Neuronal Híbrida (Embedding + Dense)

**Notebook:** [`3_models/neural_network.ipynb`](./3_models/neural_network.ipynb)

**¿Por qué una Red Neuronal con Embeddings?**
- **Aprende representaciones semánticas:** Los embeddings capturan que palabras similares están cerca geométricamente
- **Complementa TF-IDF:** Mientras TF-IDF es bag-of-words, los embeddings aprenden significados
- **Híbrida:** Combina texto (vía embeddings) con meta-features numéricas
- **Eficiente:** Arquitectura simple pero efectiva para textos cortos como tweets

**Arquitectura:**

El modelo tiene dos ramas que se combinan:

1. **Rama de Texto:**
   - TextVectorization (convierte texto a secuencias de enteros)
   - Embedding (50 dims) - vectores semánticos entrenables
   - GlobalAveragePooling1D - promedia todos los embeddings de palabras
   - Dense(64) + ReLU
   - Dropout(0.5)

2. **Rama Numérica:**
   - Meta-Features (scaled) - keywords, conteos, etc.
   - Dense(32) + ReLU
   - BatchNormalization
   - Dropout(0.5)

3. **Combinación:**
   - Concatenate - une ambas ramas
   - Dense(64) + ReLU
   - Dropout(0.5)
   - Dense(1) + Sigmoid - clasificación binaria

*Nota: Al ejecutar el notebook, se genera automáticamente el diagrama visual de la arquitectura en `neural_network_architecture.png` usando `tf.keras.utils.plot_model()`. También está [aquí](./3_models/neural_network_architecture.png) para referencia.*

**Configuración:**
- **Vocabulario:** MAX_TOKENS = 10,000
- **Longitud de secuencia:** SEQUENCE_LENGTH = 40
- **Embedding dim:** 50
- **Optimizer:** Adam (lr=0.001)
- **Loss:** Binary Crossentropy
- **Metric:** AUC
- **Callbacks:** EarlyStopping (patience=5), ReduceLROnPlateau

**Resultado:**
- **Epochs entrenados:** ~15-20 (con EarlyStopping)
- **F1-Score (Validation):** 0.7834
- **Threshold optimizado:** 0.610
- **ROC AUC:** 0.88
- **Score Kaggle:** 0.79650

**Análisis:**
- La arquitectura híbrida aprovecha tanto embeddings de texto como meta-features
- GlobalAveragePooling promedia los embeddings, capturando el "sentimiento general" del tweet
- Score de 0.79650 es sólido, pero el ensemble de stacking (0.83726) sigue siendo superior
- **Insight:** Para textos cortos como tweets, los modelos de ensemble con features diversas superan a una sola arquitectura profunda

---

## 5. Consigna Extra: Análisis ROC, Feature Importance y Matriz de Confusión (1 punto)

**Requisito:** Graficar y analizar, para alguno de los modelos de la parte III, los siguientes resultados:
- Curva ROC, explicando la selección de corte
- Feature importance
- Matriz de confusión

**Implementado en:** Todos los modelos de la Parte tienen estas visualizaciones.
- [`2_baseline/random.ipynb`](./2_baseline/random.ipynb)
- [`2_baseline/logistic_regression.ipynb`](./2_baseline/logistic_regression.ipynb)
- [`3_models/random_forest.ipynb`](./3_models/random_forest.ipynb)
- [`3_models/xgboost.ipynb`](./3_models/xgboost.ipynb)
- [`3_models/neural_network.ipynb`](./3_models/neural_network.ipynb)
- [`4_extra/four_feats.ipynb`](./4_extra/four_feats.ipynb)

Cada notebook contiene las 3 visualizaciones requeridas con análisis detallado (Excepto Random Baseline que no tiene feature importance).

---

## 6. Modelo Extra: 4 Features (1 punto)

### 6.1 Generación de las 4 Features

**Notebook:** [`4_extra/four_feats.ipynb`](./4_extra/four_feats.ipynb)

El concepto de este modelo es usar **stacking**: entrenar modelos simples y usar sus predicciones como features para un modelo final. Cada feature representa la "opinión" de un modelo diferente sobre si el tweet es un desastre.

#### Feature 1: Keyword Target Encoding (Probabilidad Histórica)

**¿Qué hace?**
- Calcula la probabilidad de que un tweet sea desastre basándose **únicamente en su keyword**
- Ejemplo: Si la keyword "fire" aparece en 100 tweets y 85 son desastres, su valor sería ~0.85

**¿Qué representa?**
- La "reputación histórica" de esa palabra clave
- Captura patrones simples: algunas keywords son fuertemente indicativas de desastres

**Implementación:**
```python
# Para cada keyword, calcula: P(desastre | keyword)
keyword_probs = train.groupby('keyword')['target'].mean()
# Smoothing para keywords raras (evita sobreajuste)
weight = count / (count + 2)
```

**Ventaja:** Simple pero efectivo para keywords muy predictivas como "wildfire", "bombing"

---

#### Feature 2: BERT Probability (Comprensión Semántica Profunda)

**¿Qué hace?**
- **BERT** (Bidirectional Encoder Representations from Transformers) lee el tweet completo
- Entiende el **contexto** y **significado** de las palabras, no solo su presencia
- Devuelve una probabilidad entre 0 y 1 de que sea desastre

**¿Qué representa?**
- La interpretación más sofisticada del texto
- Captura relaciones semánticas complejas

**Ejemplo concreto:**
```
Tweet: "Just got fired from my job 😭"
- Palabra "fired" podría confundir a modelos simples (¿fuego?)
- BERT entiende que "got fired" = despedido, NO es desastre
- Probability: ~0.15 (bajo)

Tweet: "Building on fire, people evacuating"
- BERT entiende "on fire" en contexto de edificio
- Probability: ~0.92 (alto)
```

**¿Por qué es tan bueno?**
- Pre-entrenado en Wikipedia + libros (millones de textos)
- Aprende matices del lenguaje: sarcasmo, contexto, polisemia
- Distingue "fire someone" (despedir) vs "fire" (fuego)

**Implementación:**
- DistilBERT (versión ligera y rápida de BERT)
- Fine-tuning: 2 epochs sobre nuestro dataset de tweets
- Cross-validation: 5 folds para evitar data leakage

---

#### Feature 3: Logistic Regression Probability - Words (Palabras Importantes)

**¿Qué hace?**
- **TF-IDF** identifica palabras importantes en cada tweet (descarta stop words comunes)
- **Logistic Regression** aprende qué palabras predicen desastres
- Devuelve probabilidad basada en presencia/ausencia de palabras clave

**¿Qué representa?**
- Modelo lineal simple basado en **bolsa de palabras** (bag of words)
- Captura palabras individuales y bi-gramas (pares de palabras consecutivas)

**Ejemplo:**
```
Tweet: "Massive earthquake hits California, buildings collapsed"
TF-IDF detecta palabras importantes:
- "earthquake" → peso alto positivo (indica desastre)
- "collapsed" → peso alto positivo
- "massive" → peso moderado positivo
- "hits" → peso bajo

Logistic Regression suma pesos:
→ Probability: ~0.88 (alto)
```

**Ventaja:** 
- Rápido y eficiente
- Bueno con vocabulario específico de desastres
- Complementa BERT con enfoque más simple y directo

---

#### Feature 4: Logistic Regression Probability - Chars (Patrones Ortográficos)

**¿Qué hace?**
- **TF-IDF sobre caracteres** analiza secuencias de 3-6 letras consecutivas
- Captura patrones de escritura, typos, abreviaciones

**¿Qué representa?**
- Estilo de escritura y patrones ortográficos
- Robusto ante errores de tipeo y variaciones de palabras

**Ejemplo concreto:**
```
Tweet: "OMG EARTHQUAKE!!!! everyones screaming"
N-grams de caracteres detectan:
- "earthq", "rthqua", "quake" → reconoce "earthquake" aunque tenga typo
- "!!!!!!" → patrón de urgencia/pánico
- "OMG", "everyones" (sin apóstrofe) → escritura informal/urgente

Probability: ~0.75
```

**¿Por qué es útil?**
- Tweets suelen tener typos, abreviaciones, lenguaje informal
- Captura estilo emocional: muchos signos de exclamación, CAPS LOCK
- Complementa el análisis por palabras al ser más flexible

**Diferencia con Feature 3:**
- Feature 3 (words): "earthquake" → 1 palabra
- Feature 4 (chars): "ear", "art", "rth", "thq", "hqu", "qua", "uak", "ake" → 8 patrones
- Si hay typo "eartquake", Feature 3 lo pierde, Feature 4 captura 7/8 patrones

---

### 6.1.1 ¿Cómo se Complementan las 4 Features?

Cada feature captura un aspecto diferente del tweet:

| Feature | ¿Qué Analiza? | Fortaleza | Debilidad |
|---------|---------------|-----------|-----------|
| **1. Keyword TE** | Palabra clave específica | Keywords muy predictivas | Ignora el texto |
| **2. BERT** | Contexto y semántica completa | Entiende matices del lenguaje | Computacionalmente costoso |
| **3. LR Words** | Presencia de palabras clave | Rápido, vocabulario de desastres | Pierde contexto |
| **4. LR Chars** | Patrones de escritura | Robusto a typos, estilo emocional | Menos interpretable |

**Ejemplo integrador:**
```
Tweet: "Omg there's a HUGE fire spreading fast!! #wildfire"

Feature 1 (Keyword TE): 
- keyword="wildfire" → P=0.92 ✅ (muy predictivo)

Feature 2 (BERT): 
- Lee todo el contexto: "huge", "spreading fast", tono urgente
- P=0.89 ✅

Feature 3 (LR Words):
- Detecta: "fire", "spreading", "wildfire"
- P=0.85 ✅

Feature 4 (LR Chars):
- Patrones: "!!!", "HUGE" (caps), "Omg" (informal urgente)
- P=0.78 ✅

Meta-Modelo (Random Forest):
- Combina las 4 opiniones: [0.92, 0.89, 0.85, 0.78]
- Predicción final: DESASTRE ✅ (alta confianza)
```

**Ventaja del Stacking:**
- Si BERT se confunde, las otras 3 features pueden compensar
- Si hay un typo que afecta a LR Words, BERT y LR Chars lo capturan
- El Random Forest aprende a ponderar cada feature según su confiabilidad

### 6.2 Meta-Modelo: Random Forest sobre las 4 Features

**Hiperparámetros (RandomizedSearchCV):**
- `n_estimators`: [100-500]
- `max_depth`: [3-9, None]
- `min_samples_split`: [1-5]
- `min_samples_leaf`: [1-4]
- 200 iteraciones de búsqueda

**Resultado:**
- **Mejores params:** `n_estimators=450`, `max_depth=3`, `min_samples_split=3`, `min_samples_leaf=3`
- **F1-Score (CV 5-fold):** 0.7956
- **F1-Score (Validation):** 0.7959
- **Threshold optimizado:** 0.4991 (~0.5)
- **ROC AUC:** 0.89
- **Score Kaggle:** 0.83726

**Análisis de Feature Importance:**
1. **bert_prob:** ~52% (BERT domina como esperado)
2. **lr_char_prob:** ~27% (Patrones de caracteres)
3. **lr_word_prob:** ~20% (Palabras clave)
4. **keyword_te:** ~1% (Target Encoding)

---

## 7. Comparación de Modelos

| Modelo | F1-Score (CV) | F1-Score (Val) | Threshold | ROC AUC | Score Kaggle |
|--------|---------------|----------------|-----------|---------|--------------||
| Random (Baseline) | N/A | 0.4565 | 0.5 | 0.52 | 0.51854 |
| Logistic Regression | 0.6970 | 0.7020 | 0.5538 | 0.81 | 0.72632 |
| Random Forest | 0.7248 | 0.7564 | 0.370 | 0.85 | 0.76248 |
| XGBoost | 0.7525 | 0.7698 | 0.420 | 0.87 | 0.77719 |
| Neural Network (Híbrida) | N/A | 0.7834 | 0.610 | 0.88 | 0.79650 |
| **4 Features (Stacking)** | **0.7956** | **0.7959** | **0.4991** | **0.89** | **0.83726** |


**Mejor modelo:** 4 Features Stacking (F1-Val: 0.7959, Kaggle: 0.83726)

---

## 8. Respuestas a Preguntas de la Consigna

### 8.1 Parte II: Modelo Básico (Logistic Regression)

**Pregunta:** ¿Cuál es el score obtenido en Kaggle con Logistic Regression?

**Respuesta:** **0.72632**

**Archivo de predicciones:** [`.data/submission/logistic_regression_submission.csv`](./.data/submission/logistic_regression_submission.csv)

---

### 8.2 Parte III: Modelos Avanzados

**Pregunta:** ¿Cuál es el score en la competencia para el mejor modelo de Random Forest y XGBoost?

**Respuesta:**
- **Random Forest Score Kaggle:** Pendiente (modelo mejorado en ejecución con nuevas features)
- **XGBoost Score Kaggle:** Pendiente (modelo mejorado en ejecución con nuevas features)
- **Mejor modelo general:** 4 Features Stacking con **0.83726**

**Archivos de predicciones:**
- [`.data/submission`](./.data/submission/)

**¿Por qué el 4 Features Stacking es el mejor?**
- **Diversidad de modelos:** Combina BERT (contexto semántico profundo), Logistic Regression con palabras y caracteres (patrones léxicos), y Target Encoding de keywords
- **Complementariedad:** Cada modelo captura aspectos diferentes del problema - cuando uno falla, los otros compensan
- **Meta-learning:** Random Forest aprende automáticamente cómo ponderar cada predicción según su confiabilidad
- **Regularización natural:** El ensemble reduce overfitting al promediar múltiples perspectivas

---

## 9. Conclusiones y Lecciones Aprendidas

Personalmente me llevo de este TP que entrenar modelos de machine learning efectivos requiere de tener mucho conocimiento y práctica a la hora de explorar datos, diseñar features y seleccionar modelos. No me resultó nada sencillo llegar a un modelo competitivo. En lo personal siento que muchas veces no entiendo al 100% lo que hago no tengo las heurísiticas desarrolladas para saber qué camino tomar y creo que la única vuelta es seguir practicando.

Respecto al TP en sí, la primera y 2da parte me resultaron intuitivas de realizar y entender. La parte III ya ma costó mucho más, me deje cegar por el score f1 en validation y no logré profundizar tanto como me gustaría en entender todos los hiperparámetros y en hacer un seguimietno profundo de que features estaban siendo más útiles. Termine dejando 3 modelos porque sentía que con 2 iba a estar inconpleto y al no alcanzar el objetivo de f1=0.8 en validación preferí compensarlo por este lado.

Respecto a la parte 4 nose si está OK o no hacer lo que hice... mi punto era en el feature engineering tranformar el texto en coeficientes basados en modelos simples y luego usar esos coeficientes como features para un modelo final. Me salvó muchísimo el modelo BERT ya que agarraba muchísimo el contexto del tweet y me daba una feature muy poderosa.

Apreiciaría mucho un feedback sobre mi TP porque en algún momento de la vida seguro que me vuelvo a encontrar con un problema similar y me gustaría tener unos buenos modelos y enfoques de referencia. También si es posible me gustaría conocer como otros resolvieron el TP y que criterios fueron usando para seleccionar features y modelos.

---

## Referencias y Links Útiles

- **Kaggle Competition:** [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)
- **Spacy:** [https://spacy.io/](https://spacy.io/)
- **Scikit-learn:** [https://scikit-learn.org/](https://scikit-learn.org/)
- **XGBoost:** [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
- **TensorFlow/Keras:** [https://www.tensorflow.org/](https://www.tensorflow.org/)
- **Transformers (Hugging Face):** [https://huggingface.co/docs/transformers/](https://huggingface.co/docs/transformers/)
