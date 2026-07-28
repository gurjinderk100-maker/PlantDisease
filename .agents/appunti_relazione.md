# Appunti e Insight per la Relazione Finale (MLDM UniBS)

Questo documento raccoglie in modo strutturato tutti gli appunti, le motivazioni metodologiche e le osservazioni empiriche emerse dall'analisi della repository e dallo sviluppo del progetto. Verr√† utilizzato come traccia per la stesura della relazione finale e per la preparazione del colloquio orale.

---

## 1. Mappatura Concettuale con il Materiale del Corso

| Fase della Pipeline | Concetto Teorico (Slide del Corso) | Attuazione Pratica nel Progetto |
| :--- | :--- | :--- |
| **Ingestione & Preprocessing** | `2-MLDM-DataAnalysis.pdf`, `1-Python-NumPy.pdf` | Ingestione da Kaggle API, ridimensionamento immagini a 224x224, normalizzazione dei canali RGB con media `[0.485, 0.456, 0.406]` e deviazione standard `[0.229, 0.224, 0.225]`. |
| **Data Augmentation** | `12-ANN.pdf`, `HML-Cap10-SlidesNN.pdf` | Prevenzione dell'overfitting su dataset visivi tramite rotazione casuale ($15^\circ$), flip orizzontale e crop casuale. |
| **Modello Baseline (Custom CNN)** | `12-ANN.pdf` (Architetture Convoluzionali) | Struttura a 3 blocchi convoluzionali (`Conv2d` $\to$ `BatchNorm2d` $\to$ `ReLU` $\to$ `MaxPool2d`), seguita da strati dense con `Dropout(0.5)` e `Linear`. |
| **Modello Avanzato (Transfer Learning)** | `12-ANN.pdf`, `HML-Cap10-SlidesNN.pdf` | Fine-tuning di **ResNet50** pre-addestrata su ImageNet con congelamento dei layer trasversali (backbone) e ri-addestramento della testa classificatrice finale. |
| **Valutazione e Metriche** | `Evaluation-aprile-2021-web.pdf` | Matrice di confusione, Accuracy, Precision, Recall, F1-Score pesato ed analisi degli errori di classificazione. |

---

## 2. Note Metodologiche per la Relazione Finale

- **Forma Espositiva**: La relazione deve essere redacta rigorosamente in terza persona o in forma passiva/impersonale (*"√® stato implementato"*, *"si √® osservato un incremento dell'accuracy del..."*).
- **Struttura dei Risultati**: Presentare sempre un confronto tabellare diretto tra la Rete Convoluzionale Baseline (sviluppata da zero) e la Rete con Transfer Learning (ResNet50), evidenziando il risparmio in termini di epoche di addestramento e il guadagno in accuratezza sul validation/test set.
- **Analisi delle Criticit√†**: Evidenziare la gestione delle classi sbilanciate (se presenti) e la scelta della funzione di perdita (`CrossEntropyLoss`) combinata con l'ottimizzatore `Adam` / `SGD` con learning rate adeguato.

---

## 3. Registro degli Insight e Risultati Empirici

*(Questa sezione verr√† continuamente popolata durante l'esecuzione degli esperimenti nel notebook)*

- **Insight 1 (Data Processing)**: L'uso di `DatasetFromSubset` in PyTorch garantisce che le trasformazioni di Data Augmentation vengano applicate solo al Training set, mantenendo immutati e deterministici i set di Validation e Test.
- **Insight 2 (Model Performance)**: Il modello ResNet50 in Transfer Learning converge significativamente pi√π velocemente rispetto alla Custom CNN baseline grazie alle feature a basso e medio livello gi√† estratte su ImageNet.
- **Insight 3 (Data Ingestion)**: Si √® reso necessario automatizzare l'acquisizione del dataset direttamente tramite API (Kaggle API). √à stato rilevato che per determinati repository agricoli (es. *PlantVillage*), la piattaforma Kaggle impone l'accettazione preventiva delle licenze di distribuzione (EULA), la quale richiede la verifica dell'identit√† (SMS verification) dell'utente per evitare abusi da parte dei bot. Questo passaggio garantisce la conformit√† e la riproducibilit√† etica del progetto.
- **Insight 4 (EDA e Bilanciamento)**: In fase di Analisi Esplorativa dei Dati (EDA) √® stato integrato un blocco dedicato allo studio della distribuzione delle classi e al campionamento visivo (grid 3x3) all'interno del Notebook Jupyter. La verifica del bilanciamento √® risultata cruciale prima di impostare la Data Augmentation, al fine di anticipare eventuali bias verso le patologie maggiormente rappresentate nel dataset.
- **Insight 5 (Explainable AI - Grad-CAM)**: Per superare il problema della 'black-box' tipico delle Reti Neurali, Ë stata integrata la tecnica Grad-CAM. Questo permette di estrarre le heatmap dalle attivazioni convoluzionali (es. sull'ultimo layer di ResNet50) per visualizzare empiricamente quali porzioni della foglia hanno influenzato la previsione della patologia, fornendo un alto grado di interpretabilit‡ al modello.
- **Insight 6 (Monitoraggio & TensorBoard)**: Seguendo le best practices esplorate nelle lezioni, il training loop Ë stato strumentato con SummaryWriter di PyTorch per il tracciamento in tempo reale su TensorBoard, offrendo metriche esplorabili e confrontabili in modo interattivo.
- **Insight 7 (Early Stopping)**: » stato introdotto un meccanismo custom di Early Stopping monitorando la loss di validazione, per prevenire overfitting bloccando il training e ripristinando i pesi ottimali non appena la rete inizia a perdere capacit‡ di generalizzazione (patience=3).
- **Insight 8 (Modello Ibrido Deep Learning + SVM)**: » stato costruito un modello ibrido rimuovendo l'ultimo layer di ResNet50 per utilizzarla come estrattore di feature profonde (Feature Extractor). Tali feature sono state passate a una Support Vector Machine (SVM) lineare, permettendo di combinare la potenza rappresentazionale delle CNN con il rigore dei margini di classificazione del Machine Learning classico visto a lezione.
- **Insight 9 (Visualizzazione Spazio Latente tramite t-SNE)**: Per verificare qualitativamente la capacit‡ del feature extractor di separare le patologie, lo spazio ad alta dimensionalit‡ in uscita da ResNet50 Ë stato proiettato in 2D utilizzando l'algoritmo t-SNE (t-Distributed Stochastic Neighbor Embedding). Lo scatter plot risultante dimostra l'efficacia della rete nel raggruppare (clustering intrinseco) immagini della stessa classe.
- **Insight 10 (Valutazione Avanzata Multiclasse: ROC e AUC)**: Oltre all'Accuracy e alla F1-Score, Ë stata generata l'analisi delle curve ROC multiclasse (approccio One-vs-Rest) e calcolata l'Area Under Curve (AUC), standard di valutazione statistica imprescindibile per dimostrare la robustezza del classificatore rispetto al tasso di falsi positivi.
