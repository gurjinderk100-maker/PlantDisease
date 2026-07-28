# Progetto MLDM: Design Decisions & Context (AI Assistant Guide)

Questo documento serve come **contesto di base per l'assistente IA** da utilizzare durante l'esame orale del corso di *Machine Learning e Data Mining* (Prof. Serina / Prof. Gerevini, UniBS). Contiene la cronologia di tutte le modifiche strutturali apportate al progetto originale "Plant Disease Detection", le motivazioni tecniche dietro ogni scelta (Design Decisions) e il collegamento teorico con il materiale del corso.

---

## 1. Pulizia e Preparazione Accademica
* **Azione:** Rinominato il file da `plant_desease.ipynb` a `plant_disease.ipynb` e sistemati i riferimenti nel `README.md`.
* **Perché:** La cura dei dettagli è fondamentale in ambito accademico. Errori di battitura macroscopici nel titolo del progetto minano la professionalità del lavoro.

## 2. Esportazione delle Metriche (Fondamentale per la Relazione)
* **Azione:** Introdotta la funzione `plot_and_save_history()` per generare e salvare su disco i grafici di Loss e Accuracy (sia per Train che per Validation). Aggiunto il salvataggio della *Confusion Matrix* (`plt.savefig()`).
* **Perché:** I docenti richiedono una relazione in PDF che descriva i risultati. Senza le curve di apprendimento salvate, è impossibile dimostrare visivamente la convergenza del modello e l'assenza di overfitting.

## 3. Monitoraggio Interattivo (UX & Progress)
* **Azione:** Integrata la libreria `tqdm` nei loop di addestramento (Train/Val) per avere una barra di progresso con l'ETA e l'aggiornamento della loss in tempo reale.
* **Perché:** Il training in deep learning è computazionalmente costoso. Il feedback in tempo reale dimostra maturità nella scrittura di codice Python professionale e aiuta a capire immediatamente se c'è un problema nel calcolo dei gradienti (es. loss ferma).

## 4. Riproducibilità Scientifica (Global Seeds)
* **Azione:** Aggiunta la funzione `set_seed(42)` per fissare i generatori di numeri pseudocasuali per PyTorch, NumPy e CUDA.
* **Perché:** In ambito scientifico/universitario, l'esperimento DEVE essere riproducibile. Senza un seed fisso, la Data Augmentation, i pesi casuali della Baseline CNN e il dropout genererebbero risultati diversi ad ogni esecuzione, rendendo la relazione inaffidabile.

## 5. Model Checkpointing (Salvataggio Pesi)
* **Azione:** Modificato il ciclo `train_model` affinché salvi su disco i pesi del modello alla migliore epoca (`.pth`).
* **Perché:** Salvare il modello in RAM (come era originariamente) comporta la perdita di ore di addestramento al riavvio del kernel Jupyter. Il salvataggio fisico permette di scindere la fase di addestramento da quella di inferenza (deploy).

## 6. Fine-Tuning degli Iperparametri (Learning Rate)
* **Azione:** Modificato il Learning Rate dell'ottimizzatore Adam per la ResNet50 da `0.001` a `1e-4` (`0.0001`).
* **Perché:** Questa è una *design decision* cruciale per il Transfer Learning. Usare un LR troppo alto (1e-3) su una rete pre-addestrata (ImageNet) rischia di distruggere i pesi (e le feature di basso livello) estratti in origine (fenomeno del *catastrophic forgetting*). `1e-4` è il valore canonico per il fine-tuning.

## 7. Parallelizzazione GPU (Hardware Acceleration)
* **Azione:** Inserito il costrutto `nn.DataParallel(model)` nel caso vengano rilevate GPU multiple (es. le 2x T4 di Kaggle).
* **Perché:** Dimostra competenza nell'ottimizzazione del codice per architetture distribuite, dimezzando i tempi di addestramento senza alcuna perdita di precisione matematica.

## 8. Inferenza Real-World (Test sul campo)
* **Azione:** Aggiunta la funzione `predict_image()` per testare il modello su immagini fornite da URL o caricate localmente.
* **Perché:** Un modello confinato al suo test-set è inutile. Mostrare al prof che la rete può classificare correttamente una foglia cercata su Google Immagini in quel momento (out-of-distribution) è la prova definitiva di successo.

## 9. Explainable AI (Grad-CAM)
* **Azione:** Inserita la generazione di Heatmap tramite *Gradient-weighted Class Activation Mapping* sull'immagine di test.
* **Perché:** Le CNN sono classificate come "Scatole Nere". Il Grad-CAM fornisce interpretabilità, mostrando al docente esattamente su quali pixel (es. i bordi marci della foglia) la rete si è concentrata per prendere la decisione. Scongiura il dubbio che la rete stia usando bias dello sfondo.

## 10. Modello Ibrido (Deep Learning + Machine Learning Classico)
* **Azione:** Sostituzione dell'ultimo layer di classificazione della ResNet50 per usare la rete come puro "Estrattore di Feature" ad alta dimensionalità (512-D). Le feature vengono poi passate a una **Support Vector Machine (SVM)** lineare per la classificazione finale.
* **Perché:** Design decision strategica per l'orale. Il corso verte molto sul Machine Learning classico (SVM, Random Forest). Questa ibridazione unisce le feature complesse del Deep Learning alla robustezza teorica e al calcolo dei margini della SVM vista a lezione.

## 11. Visualizzazione Spazio Latente (t-SNE)
* **Azione:** Compressione delle features 512-D in 2D tramite l'algoritmo t-SNE e plotting di uno scatter plot colorato per classe.
* **Perché:** Toccando l'argomento della riduzione della dimensionalità e dell'apprendimento non supervisionato, il grafico dimostra visivamente che la rete neurale sta effettivamente mappando foglie simili nello stesso punto dello spazio metrico.

## 12. Valutazione Accademica Avanzata (ROC / AUC Multiclasse)
* **Azione:** Generazione delle curve *Receiver Operating Characteristic* con strategia *One-vs-Rest* e calcolo dell'Area Sotto la Curva (AUC).
* **Perché:** Valutare con la sola *Accuracy* su dataset potenzialmente sbilanciati è un errore accademico classico. AUC e ROC sono il gold standard per dimostrare la solidità del classificatore e la sua sensibilità/specificità nei casi medici o fito-patologici.

---

## FAQ e Possibili Domande del Professore (Preparazione Orale)

Essendo state integrate tecniche molto avanzate per puntare al voto massimo, � altissima la probabilit� che il professore faccia domande mirate per verificare che tu abbia compreso ci� che hai implementato (e che non sia solo "copia-incolla"). Ecco come rispondere.

### Q1: "Perch� avete abbassato il Learning Rate a 1e-4 per la ResNet50?"
**Risposta Ideale:** "Perch� stavamo facendo *Transfer Learning*. La ResNet50 ha gi� imparato a riconoscere forme e contorni di base essendo pre-addestrata su ImageNet. Se avessimo usato un Learning Rate alto (es. 0.001 come nella rete creata da zero), avremmo causato il *catastrophic forgetting*, andando a distruggere violentemente quei pesi pre-addestrati. Un learning rate basso ci ha permesso di adattare delicatamente i pesi al nostro nuovo dominio (le foglie)."

### Q2: "Cos'� esattamente il t-SNE e perch� non avete usato semplicemente la PCA?"
**Risposta Ideale:** "Il t-SNE � un algoritmo di riduzione della dimensionalit� non lineare. Lo abbiamo preferito alla PCA (che � lineare) perch� le feature estratte da una rete neurale complessa giacciono su una variet� (manifold) altamente non lineare. Il t-SNE � molto pi� efficace nel mantenere vicini i punti (le foglie) che sono simili nello spazio ad alta dimensionalit�, permettendoci di verificare visivamente se la rete aveva imparato a creare cluster distinti per ogni patologia."

### Q3: "Come funziona il Modello Ibrido (CNN + SVM)? Perch� l'avete fatto?"
**Risposta Ideale:** "Volevamo unire il programma di Deep Learning a quello di Machine Learning Classico affrontato nel corso. Abbiamo tolto l'ultimo strato (Fully Connected) della ResNet50 per non fargli fare la classificazione, usandola solo come 'Estrattore di Feature'. Per ogni foglia abbiamo ottenuto un vettore di 512 numeri. Abbiamo poi dato questi vettori in pasto a una Support Vector Machine lineare. Abbiamo scoperto che le SVM si comportano in modo eccellente se alimentate con feature ad alto livello estratte da una CNN."

### Q4: "Cos'� il Grad-CAM e perch� lo ritenete utile?"
**Risposta Ideale:** "Il Grad-CAM sfrutta i gradienti calcolati all'ultimo strato convoluzionale per produrre una mappa di calore (heatmap) sull'immagine originale. Lo riteniamo essenziale per l'Interpretabilit� (Explainable AI): in ambito fito-patologico dovevamo essere certi che la rete riconoscesse davvero la *malattia* sulla foglia, e non si stesse basando su pattern sbagliati (es. il colore dello sfondo o il vaso della pianta)."

### Q5: "Come avete gestito l'Overfitting?"
**Risposta Ideale:** "Abbiamo attaccato l'overfitting su pi� fronti:
1. **Sui Dati**: Facendo Data Augmentation aggressiva (rotazioni casuali, flip) solo sul train set.
2. **Sull'Architettura**: Aggiungendo strati di Dropout (es. spegnendo il 50% dei neuroni).
3. **Sul Training**: Implementando una logica custom di *Early Stopping*. Monitoravamo la Validation Loss; se per 3 epoche consecutive non migliorava, interrompevamo il training per evitare che la rete iniziasse a memorizzare il training set."

### Q6: "Perch� avete calcolato le Curve ROC e l'AUC e non vi siete accontentati dell'Accuracy?"
**Risposta Ideale:** "L'Accuracy pu� essere ingannevole, specialmente se le classi delle malattie non sono perfettamente bilanciate. Le curve ROC (plot of True Positive Rate vs False Positive Rate) e l'integrale AUC ci danno una misura solida della capacit� del nostro classificatore multiclasse (usando un approccio One-vs-Rest) di distinguere in maniera netta i veri positivi minimizzando i falsi allarmi, che � la metrica standard in ambito diagnostico."








Ecco un documento Markdown completo e strutturato in modo accademico e professionale. È pensato per essere presentato (o usato come traccia per l'orale) a un professore esperto. Mette in luce non solo *cosa* fa il codice, ma soprattutto *perché* lo fa, dimostrando una profonda comprensione teorica e metodologica.

Puoi copiare il testo qui sotto e incollarlo in un file `.md` o in una cella Markdown del tuo notebook.

***

# Relazione Architetturale e Scelte di Design: Plant Disease Detection System

## Abstract
Il presente documento illustra l'architettura, la pipeline dati e le scelte di design implementate per lo sviluppo di un sistema di classificazione automatica delle patologie fogliari. Il progetto integra tecniche di **Deep Learning** (Custom CNN e Transfer Learning), **Machine Learning Classico** (Support Vector Machines) ed **Explainable AI** (Grad-CAM), garantendo robustezza, scalabilità e interpretabilità dei risultati.

---

## 1. Panoramica Architetturale Generale

L'architettura del sistema è stata concepita seguendo i paradigmi standard delle moderne pipeline di Computer Vision, suddivisa in quattro macro-moduli:

### A. Data Ingestion e Preprocessing
*   **Risoluzione Dinamica dei Path:** Il sistema non fa affidamento su percorsi hardcoded, ma utilizza algoritmi di esplorazione (`os.walk`) per individuare autonomamente il dataset nell'ambiente di esecuzione. Il numero di classi viene inferito dinamicamente (`len(classes)`), rendendo il codice agnostico rispetto al dataset fornito.
*   **Data Augmentation:** Applicata **esclusivamente** al Training Set (Random Crop, Flip, Rotation) per aumentare la varianza dei dati e forzare la rete ad apprendere feature invarianti, mitigando l'overfitting.
*   **Normalizzazione:** Le immagini sono normalizzate utilizzando le medie e le deviazioni standard del dataset ImageNet, requisito fondamentale per la corretta convergenza dei modelli pre-addestrati.

### B. Modellazione Deep Learning (End-to-End)
*   **Baseline (Custom CNN):** Un'architettura convoluzionale leggera creata da zero. Funge da termine di paragone (lower-bound) per valutare l'effettivo vantaggio dei modelli più complessi.
*   **Transfer Learning (ResNet50):** Utilizzo di una rete profonda pre-addestrata su ImageNet. Il *backbone* convoluzionale viene congelato (frozen) per sfruttare i filtri generici già appresi (edge detection, texture), mentre il classificatore finale (Fully Connected) viene riaddestrato sulle specifiche classi fitosanitarie.

### C. Modellazione Ibrida (CNN + SVM)
Per sfruttare il meglio di entrambi i mondi (Deep e Shallow Learning), il backbone di ResNet50 viene isolato e utilizzato come **Feature Extractor**. I vettori ad alta dimensionalità (spazio latente) estratti vengono dati in pasto a una Support Vector Machine (SVM) con kernel lineare, eccellente nel trovare iperpiani di separazione in spazi ad alta dimensionalità.

### D. Explainable AI (XAI) e Deployment
*   **Grad-CAM:** Implementato per superare il limite della "black-box" tipico delle reti neurali, generando mappe di calore che evidenziano le aree dell'immagine responsabili della classificazione.
*   **Gradio UI:** Un'interfaccia web integrata per dimostrare l'applicabilità del modello in scenari reali (inference on-the-fly).

---

## 2. Analisi delle Scelte Decisionali (Q&A)

Questa sezione anticipa le domande tecniche e metodologiche, motivando le scelte ingegneristiche adottate.

### Q1: Perché è stato utilizzato il *Global Average Pooling* (GAP) nella Custom CNN al posto della classica operazione di *Flattening*?
**Risposta:** L'uso del `Flatten` seguito da strati `Linear` (Fully Connected) genera un numero enorme di parametri addestrabili, rendendo la rete estremamente prona all'**overfitting**, specialmente su dataset di medie dimensioni. Inoltre, vincola la rete a una specifica dimensione spaziale di input (es. 224x224). 
Sostituendo il Flatten con un `AdaptiveAvgPool2d((1, 1))`, si riduce ogni mappa di feature a un singolo valore (la media spaziale). Questo approccio:
1. Abbassa drasticamente la complessità computazionale (milioni di parametri in meno).
2. Rende l'architettura **invariante alle traslazioni spaziali** e alla risoluzione dell'immagine in ingresso.

### Q2: Nella pipeline ibrida (CNN + SVM), come è stato gestito il rischio di *Data Leakage*?
**Risposta:** Il Data Leakage è un errore metodologico grave che si verifica quando informazioni del set di validazione o test "inquinano" la fase di addestramento. 
Per evitarlo, l'estrazione delle feature per l'addestramento della SVM è stata eseguita su un **sottoinsieme del Training Set**, e *non* sul Validation Set. Il Validation Set è stato mantenuto rigorosamente isolato per l'ottimizzazione degli iperparametri, mentre la valutazione finale della SVM è avvenuta esclusivamente sul Test Set.

### Q3: Perché applicare il Transfer Learning congelando i pesi (Freezing) anziché fare un Fine-Tuning completo di tutta la rete?
**Risposta:** ResNet50 ha oltre 23 milioni di parametri. Un fine-tuning completo su un dataset specifico come *PlantVillage* comporterebbe due rischi:
1. **Overfitting massiccio**, poiché la rete memorizzerebbe il dataset di training.
2. **Catastrophic Forgetting**, ovvero la distruzione dei filtri di basso livello (bordi, colori, gradienti) faticosamente appresi su ImageNet.
Congelando il backbone, si utilizza la rete come un estrattore di feature "robusto e generalista", limitando l'aggiornamento dei gradienti solo agli ultimi strati densi (classificatore), garantendo una convergenza rapida e stabile.

### Q4: Quali strategie sono state implementate per prevenire l'Overfitting?
**Risposta:** L'overfitting è stato affrontato su più livelli architetturali:
*   **A livello di Dati:** Data Augmentation (rotazioni, flip e crop casuali) per impedire alla rete di memorizzare l'orientamento esatto delle foglie.
*   **A livello di Architettura:** Inserimento di layer di **Dropout** (con probabilità 0.4 - 0.5) prima dei classificatori finali per forzare la ridondanza dei neuroni, e utilizzo della **Batch Normalization** per regolarizzare le attivazioni interne.
*   **A livello di Addestramento:** Implementazione dell'**Early Stopping** (interruzione del training se la Validation Loss non migliora per $N$ epoche) e utilizzo di **Learning Rate Schedulers** (`ReduceLROnPlateau`) per far convergere il modello in modo più dolce nei minimi locali della funzione di costo.

### Q5: Qual è il valore aggiunto dell'utilizzo di Grad-CAM in questo specifico dominio applicativo?
**Risposta:** Nel settore Agritech, l'affidabilità (Trust) è fondamentale. Una rete neurale potrebbe raggiungere un'alta accuratezza "barando", ovvero imparando correlazioni spurie (es. classificando una malattia non guardando la foglia, ma il colore dello sfondo o il vaso). 
Grad-CAM (Gradient-weighted Class Activation Mapping) calcola i gradienti della classe target rispetto all'ultima mappa convoluzionale. Questo ci permette di verificare visivamente che il modello stia effettivamente "guardando" le macchie necrotiche o i pattern patologici sulla foglia, validando la robustezza semantica del modello.

### Q6: Perché utilizzare la t-SNE per visualizzare lo spazio latente?
**Risposta:** Le feature estratte dalla ResNet50 risiedono in uno spazio a 2048 dimensioni, impossibile da interpretare per un essere umano. La **t-SNE** (t-Distributed Stochastic Neighbor Embedding) è una tecnica di riduzione della dimensionalità non lineare che preserva le distanze locali. Proiettando queste feature in 2D, possiamo verificare se la rete ha imparato a separare correttamente le classi fitosanitarie in cluster distinti *prima* ancora che intervenga il classificatore finale, dimostrando l'efficacia del processo di feature extraction.

Conclusione
Se vuoi la perfezione, assicurati di usare la versione completa di PlantVillage (38 classi, 54.000 immagini), nello specifico il dataset **`abdallahalidev/plantvillage-dataset`** disponibile su Kaggle, utilizzando esclusivamente la sottocartella **`color`**. Questa è considerata lo standard accademico (rispetto alle versioni ridotte o in scala di grigi).
Ma non impazzire a cercare dataset da centinaia di gigabyte. L'eccellenza in questi esami non si valuta da quanti dati hai scaricato, ma da come li hai trattati, come hai addestrato il modello, e soprattutto da quanto sei consapevole dei limiti del tuo stesso sistema.
