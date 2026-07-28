# Programma Completo e Dettagliato del Corso (MLDM - UniBS)

Questo file costituisce la guida di riferimento ufficiale per l'IA. Qualsiasi modello, algoritmo, preprocessamento o tecnica utilizzata nei notebook o nella relazione DEVE essere contenuta in questo elenco o direttamente riconducibile ad esso.

---

## Macro-Modulo 1: Introduzione e Fondamenti di Data Science & Python

### 1.1 Introduzione al Machine Learning e Data Mining
- **Concetti Base**: Definizione di apprendimento automatico, tassonomia dell'apprendimento (Supervisionato, Non-Supervisionato, Semi-Supervisionato, Apprendimento per Rinforzo).
- **Pipeline di Progetto**: Formulazione del problema, ciclo di vita dei dati, addestramento, validazione e test.
- **Riferimenti**: `1-introduction2023.pdf`, `Kaggle.pdf`.

### 1.2 Programmazione in Python & R per Data Science
- **Python Scientifico**: Strutture dati native, manipolazione avanzata.
- **NumPy**: Vettori, matrici, ndarray, operazioni vettorializzate, broadcasting, indexing e slicing (`1-Python-NumPy.pdf`).
- **Pandas**: Series e DataFrame, gestione missing values, filtering, `groupby`, merging, `astype('category')`, `describe()`, `dtypes` (`05a-pandas-titanic Extended.ipynb`).
- **Data Visualization**: Matplotlib, Seaborn e ggplot (`geom_point`, istogrammi, boxplot, scatterplot, matrici di correlazione).
- **R Introduction**: Sintassi base e librerie in R (`Lab01-Rintro.ipynb`).

---

## Macro-Modulo 2: Data Preprocessing & Exploratory Data Analysis (EDA)

### 2.1 Analisi Esplorativa dei Dati (EDA)
- Statistiche descrittive (media, mediana, deviazione standard, percentili, correlazioni).
- Analisi dello sbilanciamento delle classi (Imbalanced Datasets).

### 2.2 Preprocessing e Feature Engineering
- **Missing Values**: Imputazione con media, mediana, moda o dropping.
- **Feature Scaling**: Min-Max Scaling ($[0,1]$), Standardizzazione ($Z$-score scaling con media 0 e varianza 1).
- **Categorical Encoding**: One-Hot Encoding, Label Encoding, gestione tipo dati ordinali/nominali.
- **Riferimenti**: `2-MLDM-DataAnalysis.pdf`, `Data_analysis_and_preprocessing.ipynb`.

---

## Macro-Modulo 3: Apprendimento Supervisionato (Supervised Learning)

### 3.1 Concept Learning e Apprendimento Induttivo
- Spazio delle ipotesi, algoritmo **FIND-S**, algoritmo **Candidate Elimination** (General & Specific Boundaries, Version Space).
- **Riferimenti**: `2-Concept_Learning.pdf`.

### 3.2 K-Nearest Neighbors (KNN)
- Algoritmo instance-based / lazy learning per classificazione e regressione.
- Metriche di distanza (Euclidea, Manhattan, Minkowski), scelta di $K$.
- **Riferimenti**: `KNN-code.ipynb`.

### 3.3 Alberi di Decisione (Decision Trees)
- Struttura dell'albero (nodi di split, nodi foglia).
- Criteri di split: Information Gain, Entropia (ID3/C4.5), Indice di Gini (CART).
- Overfitting e tecniche di potatura (*pruning*: pre-pruning e post-pruning, profondità massima).
- **Riferimenti**: `3-Basic_classification-Decision_Trees.pdf`, `4-Decision_Trees_Overfitting-nonotes.pdf`.

### 3.4 Support Vector Machines (SVM)
- Iperpiano a margine massimo, vettori di supporto.
- Soft Margin e parametro di regolarizzazione $C$.
- Kernel Trick: Kernel Lineare, Polinomiale, RBF (Radial Basis Function), Sigmoide.
- **Riferimenti**: `MLDM-Support-Vector-Machines.pdf`, `05_support_vector_machines.ipynb`.

### 3.5 Ensemble Learning
- **Bagging (Bootstrap Aggregating)**: **Random Forest** (selezione casuale delle feature ad ogni split).
- **Boosting**: Apprendimento sequenziale (AdaBoost, Gradient Boosting, XGBoost).
- **Riferimenti**: `MLDM-Ensemblelearning.pdf`.

---

## Macro-Modulo 4: Apprendimento Bayesiano (Bayesian Learning)

- Teorema di Bayes: a priori, likelihood, a posteriori, ipotesi MAP (Maximum A Posteriori) e ML (Maximum Likelihood).
- Classificatore **Naive Bayes** (Gaussiano, Multinomiale, Bernoulli) ed assunzione di indipendenza condizionale.
- Introduzione alle Reti Bayesiane (DAG).
- **Riferimenti**: `BayesLearning-web.pdf`.

---

## Macro-Modulo 5: Reti Neurali Artificiali & Deep Learning (ANN & DL)

### 5.1 Perceptron e Multilayer Perceptron (MLP)
- Perceptron di Rosenblatt e limite di separabilità lineare (XOR).
- MLP e strati nascosti (hidden layers).
- Funzioni di attivazione: Sigmoide, Tanh, **ReLU**, Leaky ReLU, Softmax.
- Algoritmo di **Backpropagation** (Chain Rule) ed ottimizzatori (SGD, Adam).
- **Riferimenti**: `12-ANN.pdf`, `HML-Cap10-SlidesNN.pdf`, `XOR_keras.ipynb`, `MLP from scratch.ipynb`.

### 5.2 Regolarizzazione ed Addestramento di Reti Profonde
- Dropout, Weight Decay ($L_1$/$L_2$), Batch Normalization.
- Early Stopping, checkpoint dei pesi, TensorBoard.
- **Riferimenti**: `11_training_deep_neural_networks_activations_regularization.ipynb`, `Tensorboard_keras.ipynb`.

### 5.3 Reti Neurali Convoluzionali (CNN) & Transfer Learning
- Strati Convoluzionali (`Conv2d`), Kernel, Stride, Padding.
- Layer di Pooling (`MaxPool2d`, Average Pooling).
- Architetture: LeNet-5, VGG, ResNet.
- **Transfer Learning & Fine-Tuning**: Reti pre-addestrate su ImageNet (es. ResNet50), congelamento della backbone (`requires_grad = False`), sostituzione della testa FC.
- **Riferimenti**: `LeNet-5 TensorFlow.ipynb`, `PyTorch_CNN_FMNIST.ipynb`, `H7_cnn_transfer_learning.ipynb`, `plant_desease.ipynb`.

### 5.4 Sequenze, NLP e Modelli Avanzati
- RNN, LSTM per serie temporali e testo.
- NLP: Bag of Words, TF-IDF, Word Embeddings, Attention & Transformer Encoder-Decoder (BERT).
- Object Detection: YOLOv5.
- **Riferimenti**: `Natural-Language-Processing-2024.pdf`, `15_processing_sequences_using_rnns_and_cnns_Torch.ipynb`, `Sentiment_Bert.ipynb`, `H7- YOLOv5_exercise.ipynb`.

---

## Macro-Modulo 6: Apprendimento Non Supervisionato e Clustering

### 6.1 Algoritmi di Clustering
- **K-Means**: Inizializzazione, assegnazione iterativa, aggiornamento centroidi. Scelta di $K$ tramite Elbow Method e Silhouette Score.
- **Clustering Gerarchico**: Agglomerativo e Divisivo, dendrogrammi, linkage (Single, Complete, Average, Ward).
- **DBSCAN**: Clustering basato su densità (Core points, Border points, Noise).
- **Riferimenti**: `MLDM-Cluster_analysis-1.pdf`, `MLDM-Cluster_analysis-5.pdf`, `MLDM-Cluster_analysis-7.pdf`, `09-K_Means from Scratch.ipynb`.

---

## Macro-Modulo 7: Valutazione dei Modelli e Metriche (Model Evaluation)

- **Partizionamento Dati**: Train Set, Validation Set, Test Set, $K$-Fold Cross-Validation.
- **Metriche per la Classificazione**:
  - Matrice di Confusione (TP, FP, TN, FN).
  - Accuracy, Precision, Recall, F1-Score (Macro, Micro, Weighted).
  - Curva ROC e area sotto la curva (ROC-AUC).
- **Metriche per il Clustering**: Inertia (In-cluster sum of squares), Silhouette Score.
- **Riferimenti**: `Evaluation-aprile-2021-web.pdf`.
