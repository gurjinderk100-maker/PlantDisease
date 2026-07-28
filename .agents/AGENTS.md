# Regole Globali del Progetto (Project-Scoped Rules)

## 1. Documentazione di Riferimento del Progetto
L'IA deve fare costante riferimento ai seguenti file guida contenuti nella cartella `.agents`:
- **[programma_corso.md](file:///192.168.1.6/Progetto_MachineLearning_and_DataMining/.agents/programma_corso.md)**: Il sillabo completo di tutti gli argomenti d'esame.
- **[linee_guida_esame.md](file:///192.168.1.6/Progetto_MachineLearning_and_DataMining/.agents/linee_guida_esame.md)**: Modalità d'esame, criteri di consegna e strategie per ottenere 30/30.
- **[standard_codice_e_riproducibilita.md](file:///192.168.1.6/Progetto_MachineLearning_and_DataMining/.agents/standard_codice_e_riproducibilita.md)**: Standard per la riproducibilità del codice Python/Jupyter e gestione dinamica hardware/Google Colab.
- **[appunti_relazione.md](file:///192.168.1.6/Progetto_MachineLearning_and_DataMining/.agents/appunti_relazione.md)**: Registro automatico degli insight e spunti per la relazione finale.

---

## 2. Stile di Scrittura (Relazioni, Documentazione e Commenti nel Codice)
- **Registro Linguistico Impersonale ed Accademico**: Nelle relazioni (in LaTeX o Markdown), nella documentazione e nei commenti al codice, **non utilizzare mai la prima persona plurale** ("noi abbiamo fatto", "usiamo", "abbiamo scelto", ecc.).
- **Forma Passiva o Implicita**: Utilizzare sempre forme impersonali o passive (es. *"è stato applicato"*, *"viene utilizzato"*, *"il modello è stato addestrato mediante"*, *"si osserva che"*).
- **Riferimento Scientifico e Tecnico**: Non fare mai riferimento esplicito alle slide di lezione o al professore nella relazione finale (es. non scrivere *"come spiegato nelle slide"* o *"come fatto a lezione"*). I riferimenti devono basarsi esclusivamente sulla letteratura scientifica e su standard di dominio ben consolidati.

---

## 3. Aderenza al Programma del Corso (MLDM UniBS - Prof. Serina / Prof. Gerevini)
- **Scope dei Metodi e degli Algoritmi**: Qualsiasi tecnica, modello, metrica o algoritmo utilizzato deve rientrare nel programma ufficiale del corso (esplicitato in `programma_corso.md`):
  - **Preprocessing & Data Analysis**: Imputazione missing values, Feature Scaling (StandardScaler, MinMaxScaler), Encoding (One-Hot, Label Encoding), EDA con Pandas/Seaborn/Matplotlib.
  - **Supervised Learning**: Decision Trees, Random Forest / Ensemble Learning, Support Vector Machines (SVM), Reti Neurali (MLP, CNN, Transfer Learning con ResNet/VGG).
  - **Unsupervised Learning**: Clustering (K-Means, Hierarchical Clustering, DBSCAN).
  - **Evaluation & Metrics**: Matrice di confusione, Accuracy, Precision, Recall, F1-Score, ROC-AUC, Cross-Validation.
- **Divieto di Overengineering Fuori Corso**: Non introdurre modelli o architetture esotiche non trattate a lezione se non adeguatamente giustificate come estensioni della baseline di corso.

---

## 4. Formattazione dei Notebook Jupyter (`.ipynb`)
- **Stile e Struttura del Professore**: I notebook devono ricalcare lo stile pulito e didattico dei notebook di laboratorio (`mldmlab` / `GDrive_Colaboratory`), seguendo la struttura descritta in `standard_codice_e_riproducibilita.md`:
  1. **Header Markdown e Titoli di Sezione**: Utilizzare titoli chiari in Markdown/HTML (`# Titolo`, `## 1. Setup e Ingestione Dati`, `## 2. EDA`, ecc.).
  2. **Spiegazioni Testuali tra le Celle**: Ogni blocco di codice significativo deve essere preceduto da una cella Markdown che spiega lo scopo dell'operazione.
  3. **Struttura a Pipeline**: Setup, EDA, Preprocessing, Baseline, Advanced Model, Training/Validation Loop, Evaluation & Metrics.

---

## 5. Gestione degli Appunti per la Relazione (`.agents/appunti_relazione.md`)
- **Salvataggio Automatico degli Insight**: Durante lo sviluppo del codice o la risoluzione di criticità, l'IA deve annotare i punti chiave all'interno del file **[.agents/appunti_relazione.md](file:///192.168.1.6/Progetto_MachineLearning_and_DataMining/.agents/appunti_relazione.md)**.

---

## 6. Esecuzione e Ambiente (Kaggle/Colab)
- **Esecuzione Esclusivamente in Cloud**: Tutto il codice (in particolare il training dei modelli, l'ingestione dei dataset e l'inferenza) dovrà essere concepito, testato ed eseguito esclusivamente sulle piattaforme cloud Kaggle o Google Colab. **Non** si devono scaricare dataset (es. tramite script python) né eseguire script di addestramento pesanti in locale sul file system. Ogni istruzione o blocco di codice generato deve presupporre un ambiente notebook in cloud.
