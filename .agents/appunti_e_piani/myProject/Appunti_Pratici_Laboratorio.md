# Appunti Pratici di Laboratorio - Machine Learning & Data Mining
*Raccolta delle best practices, librerie e metodi utilizzati dal Prof. Serina/Gerevini durante le esercitazioni, utile come linea guida per lo sviluppo del progetto finale.*

---

## 1. Architettura Standard di un Progetto (Il "Metodo del Prof")
Analizzando tutti i file `.ipynb` svolti a lezione, emerge una struttura ricorrente e rigorosa che dobbiamo replicare nel nostro progetto su Colab per dimostrare di aver assimilato il metodo di lavoro. La pipeline standard è:

1. **Setup & Imports**: Configurazione dell'ambiente (Colab/Kaggle) e importazione di tutte le librerie.
2. **Load Data**: Caricamento del dataset, spesso tramite Pandas o torchvision `DataLoader`.
3. **Data Analysis and Preprocessing (EDA)**: Studio dei dati. Fase critica che il prof chiama spesso "Feature engineering phase".
4. **Training (Modelli Base vs Complessi)**: "You can consider different models splitting in train/val". Partire sempre da una baseline (es. SVM o Decision Tree) prima di passare alle Reti Neurali.
5. **Parameter Tuning**: Ricerca dei migliori iperparametri (es. GridSearchCV o Optuna).
6. **Evaluate accuracy**: Valutazione rigorosa del modello.
7. **Submit the final model / Conclusion**: (Rifacendosi alle sfide Kaggle affrontate a lezione).

---

## 2. Le Librerie Fondamentali (Stack Tecnologico)
Basandomi sulla frequenza di import nei notebook del corso, lo stack che dobbiamo dominare e giustificare all'esame è:

- **Scikit-Learn (`sklearn`)**: È la libreria dominante in assoluto (centinaia di utilizzi). Usata per *tutto* il pre-processing e il machine learning classico.
- **Visualizzazione (`matplotlib` e `seaborn`)**: Obbligatorio l'uso di queste due per mostrare i grafici (Loss, Accuracy, Matrice di confusione).
- **Manipolazione Dati (`numpy` e `pandas`)**: Le fondamenta per manipolare tensori, array e dataset tabulari.
- **Deep Learning (`tensorflow/keras` e `torch`)**: Il professore ha mostrato entrambi i framework! Ha usato estesamente le architetture `Sequential` (Keras) ma ha dedicato molti laboratori anche a PyTorch (`DataLoader`, `torch.nn`). Possiamo scegliere liberamente quello con cui ci troviamo meglio per il Transfer Learning, la scelta di PyTorch è perfettamente allineata.

---

## 3. I Metodi e le Classi Più Usati (Da sapere all'orale)
Ecco le funzioni specifiche che il prof ha usato più spesso e che **dobbiamo** includere (o saper spiegare) nel nostro progetto Plant Disease:

### A. Data Split e Preprocessing
- `train_test_split`: Sempre usato per dividere rigorosamente il dataset in training e test set, per evitare il *data leakage*.
- `StandardScaler`: Quasi tutti i notebook di classificazione classica scalano i dati. Per le immagini (nel nostro progetto), useremo l'equivalente `transforms.Normalize` di PyTorch, che concettualmente fa la stessa cosa: porta i dati in un range matematicamente digeribile dalla rete.

### B. Machine Learning Classico (Le Baseline)
- `SVC` (Support Vector Classifier): Il classificatore classico più usato dal prof. Potremmo usarlo come modello "scadente" iniziale per dimostrare *perché* le CNN sono necessarie per le foglie.
- `DecisionTreeClassifier` e `RandomForestClassifier`: Molto presenti. Ottimi per spiegare l'importanza delle feature, ma meno adatti alle immagini raw.
- `KMeans`: Algoritmo principe per l'Apprendimento Non Supervisionato. 

### C. Deep Learning
- `Sequential` / `nn.Sequential`: Il modo standard insegnato a lezione per impilare i layer delle reti neurali (Dense/Linear, Conv2D, Dropout).
- `DataLoader`: Il modo corretto, insegnato nei lab di PyTorch, per caricare i dati a "lotti" (batch) senza saturare la RAM, essenziale per il nostro dataset di migliaia di immagini di foglie.

### D. Valutazione (Evaluation)
- `accuracy_score`: La metrica di base.
- `confusion_matrix`: Il prof la usa quasi sempre per mostrare visualmente dove il modello si "confonde". Nel nostro caso, mostrerà se il modello confonde due malattie simili della pianta di pomodoro.

---

## 4. Regole d'Oro per il Progetto (Come ambire al 30)
Mettendo insieme quanto visto:
1. **Mai saltare l'EDA**: Il professore dà molta importanza alla "Data Analysis". Dobbiamo stampare a video qualche immagine del dataset originale e mostrare il bilanciamento delle classi prima di dare in pasto le immagini alla CNN.
2. **Validazione Rigorosa**: Non mostrare solo l'accuracy di training (rischio overfitting). Dobbiamo plottare le curve affiancate (Train vs Validation).
3. **Hyperparameter Tuning**: Dimostriamo di non aver scelto il Learning Rate o il numero di Epoche a caso, ma facendo dei test (anche solo citandoli nella relazione).
4. **Notebook Pulito**: I notebook del prof sono divisi in sezioni chiare con intestazioni Markdown (es. "Load Data", "Build the model"). Dobbiamo mantenere lo stesso ordine.
