# Progetto Machine Learning e Data Mining - Plant Disease Detection

Questo repository contiene il progetto finale per il corso di Machine Learning e Data Mining. L'obiettivo è riconoscere in automatico le malattie delle piante a partire da foto delle foglie, usando tecniche di Deep Learning.

Tutto il codice è stato diviso in due notebook principali. Sono stati pensati per girare direttamente su cloud (es. Kaggle) per poter usare le GPU gratuitamente ed evitare ore di calcolo sul computer locale.

## Cosa c'è nella cartella
Qui si trovano i file essenziali per la consegna:
- `Progetto_MLDM_Relazione.pdf`: La relazione finale con analisi, scelte progettuali e risultati.
- `Progetto_MLDM_Training.ipynb`: Il notebook pesante. Contiene il caricamento del dataset PlantVillage, il preprocessing, l'addestramento della CNN base e della ResNet50 (Transfer Learning), fino alla valutazione con la SVM.
- `Progetto_MLDM_Inferenza_XAI.ipynb`: Un notebook leggero da usare se non si vuole rifare il training. Carica i pesi salvati e lancia un'interfaccia web per testare le immagini al volo usando Grad-CAM.
- `relazione_latex/`: Il codice sorgente usato per compilare la relazione.
- `resnet50_best.pth`: I pesi del modello ResNet50 addestrato, da usare in caso non si voglia rifare il training.

## Istruzioni per l'uso

I notebook sono testati principalmente su Kaggle. Ecco come farli partire.

### Opzione 1: Addestramento da zero (Training)
Se si ha bisogno di riaddestrare i modelli (ci vorrà circa un'ora):
1. Andare su Kaggle, creare un nuovo notebook e importare `Progetto_MLDM_Training.ipynb`.
2. Attivare la GPU dal pannello a destra (Accelerator > GPU T4 x2).
3. Aggiungere i dati: cliccare su "Add Data", cercare `abdallahalidev/plantvillage-dataset` e aggiungerlo. Il codice è già impostato per pescare dalla cartella `color`.
4. Eseguire il codice. Alla fine verranno salvati i grafici e i file dei pesi addestrati (come `resnet50_best.pth`) nella cartella di output di Kaggle.

### Opzione 2: Test veloce e Interfaccia Web (Inferenza)
Se si vuole solo provare la rete neurale e la UI senza aspettare il training:
1. Importare su Kaggle il notebook `Progetto_MLDM_Inferenza_XAI.ipynb`.
2. Caricare i pesi del modello (`resnet50_best.pth` ottenuto dall'Opzione 1).
3. Assicurarsi che l'opzione "Internet" sia accesa nelle impostazioni a destra di Kaggle (serve per far andare Gradio).
4. Eseguire tutto. Alla fine comparirà un'interfaccia interattiva dove si potrà trascinare le foto delle foglie e vedere la malattia diagnosticata, oltre alla mappa di calore (Grad-CAM) che mostra dove la rete ha guardato.

## Dettagli aggiuntivi
- Dataset: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- Il dataset non è stato caricato nel repo GitHub o nello zip per evitare di appesantire inutilmente il file, su Kaggle è già presente la versione ufficiale pronta all'uso.
- Per spiegazioni dettagliate sui blocchi di codice, consultare il capitolo 11 della relazione.
