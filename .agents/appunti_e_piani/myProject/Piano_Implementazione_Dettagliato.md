# Progetto Machine Learning e Data Mining - Piano di Implementazione Dettagliato

## Contesto e Analisi del Materiale
Questo documento rappresenta il piano d'azione per la realizzazione del progetto finale per il corso di Machine Learning e Data Mining (MLDM). L'obiettivo primario è strutturare un lavoro di alta qualità ingegneristica per ambire al voto massimo (30/30).

Analizzando il materiale del corso fornito nelle directory `Materiale_del_corso_slide` e `Materiale_del_corso_esercizi`, il piano si integra con i seguenti macro-argomenti e documenti:
1. **Introduzione, Python e Colab**: Riferimenti a `1-Python-Intro.pdf`, `1-Python-NumPy.pdf`, e `Kaggle.pdf`. L'uso di Google Colab è perfettamente in linea con quanto trattato.
2. **Data Analysis ed EDA (Exploratory Data Analysis)**: Riferimento a `2-MLDM-DataAnalysis.pdf`.
3. **Valutazione Modelli e Metriche**: Riferimento a `Evaluation-aprile-2021-web.pdf` (Accuracy, Precision, Recall, Matrice di Confusione).
4. **Apprendimento Supervisionato (Reti Neurali)**: Riferimento a `12-ANN.pdf` e `HML-Cap10-SlidesNN.pdf`.
5. **Apprendimento Non Supervisionato (Clustering)**: Riferimento a `MLDM-Cluster_analysis-1.pdf`, `MLDM-Cluster_analysis-5.pdf`, `MLDM-Cluster_analysis-7.pdf`.

Tra le due proposte descritte in `implementation_plan_original.md` (Plant Disease vs Skincare & Armocromia), **è stato scelto in via definitiva di procedere con il progetto "Plant Disease Detection"**, come concordato e confermato.
*Motivazione accademica*: Questo progetto permette di esplorare in modo approfondito l'Apprendimento Supervisionato e la Computer Vision (Reti Neurali Convoluzionali e Transfer Learning), offrendo un focus ingegneristico più solido, con minor incertezza sui dataset (PlantVillage è uno standard garantito) e garantendo alte probabilità di successo e valutazione massima (30/30).

## Fasi Successive (Implementazione Plant Disease)

Una volta confermato il progetto, verranno eseguite in automatico le seguenti fasi, e tutti i file verranno generati all'interno della cartella `myProject`:

### Fase 1: Sviluppo del Codice (Colab)
- Completamento del notebook `plant_desease.ipynb` contenente le istruzioni esatte per l'addestramento su Google Colab.
- Implementazione del codice Python (diviso in celle) per l'EDA, il Preprocessing (Data Augmentation), la Custom CNN Baseline e il modello avanzato in Transfer Learning (ResNet50), con riferimenti chiari a come questi concetti si mappano sulle slide del corso (`12-ANN.pdf` e `HML-Cap10-SlidesNN.pdf`).

### Fase 2: Relazione in LaTeX / Markdown
- Verrà creata la struttura completa della relazione finale nella cartella `myProject/relazione_latex/` (o file Markdown).
- La relazione conterrà paragrafi preimpostati, sezioni metodologiche (strutturate come da linee guida) e linguaggio accademico adatto a studenti di Ingegneria Magistrale, pronta per essere riempita con i grafici generati da Colab (Confusion Matrix, Loss/Accuracy curves).

### Fase 3: Cartella "concetti" per la preparazione all'Esame Orale
- Verrà creata la directory `myProject/concetti/` contenente file Markdown dedicati a spiegare (in modo semplice, per chi parte da zero, ma con il rigore richiesto dal prof) i concetti chiave per affrontare il colloquio orale:
  - `01_Reti_Neurali_e_CNN.md` (rif. `12-ANN.pdf`)
  - `02_Transfer_Learning.md`
  - `03_Overfitting_DataAugmentation_e_Valutazione.md` (rif. `Evaluation-aprile-2021-web.pdf`)
