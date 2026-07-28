# Progetto: Plant Disease Detection con Deep Learning

Il fatto che tu voglia usare **Google Colab** è un'ottima scelta strategica: ci eviterà qualsiasi problema di compatibilità sul tuo Mac Pro e ci darà accesso a GPU potenti direttamente dal cloud, essenziali per addestrare reti neurali in tempi brevi.

> [!NOTE]
> **Che cos'è Kaggle?**
> Immagina Kaggle come il "GitHub" degli scienziati dei dati e dell'Intelligenza Artificiale. Acquisita da Google, è la piattaforma più grande al mondo dedicata al Machine Learning. Mette a disposizione gratuitamente enormi quantità di **Dataset** (raccolte di dati pronte per essere analizzate, nel nostro caso migliaia di foto di foglie catalogate) ed è la fonte standard accademica e industriale per reperire dati affidabili su cui addestrare i modelli. Tutti i tuoi colleghi (e ricercatori di tutto il mondo) prendono i dati da lì.

Di seguito troverai la proposta formale da inviare al professore e la scaletta della relazione.

---

## 1. Proposta Formale da Inviare al Professore

*Copia e incolla questo testo per l'approvazione. È formulato con un linguaggio accademico adatto a una laurea magistrale, evidenziando che coprirai l'intero ciclo di vita di un progetto di Machine Learning.*

**Oggetto:** Proposta Progetto ML&DM - Plant Disease Detection tramite Reti Neurali Convoluzionali e Transfer Learning

Gentile Professore,

Vorrei sottoporle la mia proposta di progetto per il corso di Machine Learning e Data Mining. Prendendo spunto da un progetto di sensoristica per piante che avevo sviluppato in passato, vorrei affrontare il problema del riconoscimento automatico delle patologie fogliari a partire da immagini (Image Classification). 

L'obiettivo del progetto è confrontare le performance di approcci di base con architetture avanzate di Deep Learning.

**Dataset:** Utilizzerò il *PlantVillage Dataset* (o analogo *Plant Disease Dataset* disponibile su Kaggle), che offre decine di migliaia di immagini etichettate divise in classi di piante sane e patologiche.

**Metodologia e Tecnologie (Python, PyTorch):**
1. **Data Pre-processing ed EDA**: Analisi della distribuzione delle classi, ridimensionamento e normalizzazione delle immagini. Verranno applicate tecniche di *Data Augmentation* per prevenire l'overfitting.
2. **Baseline Model**: Sviluppo di un modello di riferimento (es. una semplice CNN Custom o l'estrazione di feature visive date in pasto a un classificatore classico come Random Forest/SVM).
3. **Modello Avanzato (Transfer Learning)**: Implementazione di Reti Neurali Convoluzionali profonde pre-addestrate (es. ResNet50 o EfficientNet), specializzandole sul dominio botanico tramite fine-tuning.
4. **Valutazione**: Confronto rigoroso dei modelli tramite Accuracy, Precision, Recall, F1-Score, curve di apprendimento e Matrice di Confusione sul Test Set.

Il tutto sarà sviluppato in ambiente Google Colab per sfruttare l'accelerazione hardware. 
Resto in attesa di un Suo riscontro o di eventuali suggerimenti per calibrare al meglio il lavoro.

Cordiali saluti.

---

## 2. Scaletta della Relazione (da far approvare al Professore)

*Questa scaletta dimostra al professore che affronterai il problema in modo scientifico e strutturato. Include concetti, tecnologie e le fasi di costruzione del software.*

1. **Introduzione e Definizione del Problema**
   - Contesto applicativo: l'importanza del riconoscimento precoce delle malattie delle piante.
   - Obiettivo del progetto (classificazione multiclasse).
2. **Esplorazione e Analisi dei Dati (EDA)**
   - Presentazione del Dataset di Kaggle (dimensione, numero di classi).
   - Analisi del bilanciamento delle classi (se ci sono più foto per certe malattie rispetto ad altre).
   - Visualizzazione di alcuni sample dal dataset.
3. **Pipeline di Pre-processing e Tecnologie Usate**
   - **Tecnologie**: Python, ambiente Google Colab, PyTorch (framework per Deep Learning), Scikit-Learn.
   - Suddivisione dei dati in *Training*, *Validation* e *Test Set*.
   - *Data Augmentation*: come abbiamo manipolato le immagini (rotazioni, zoom) in memoria per rendere la rete neurale più robusta.
4. **I Modelli di Machine Learning Implementati (Passi di Costruzione)**
   - **Passo 1: Definizione della Baseline**. Costruzione di un modello semplice (es. una CNN a 3 strati) come punto di riferimento minimo.
   - **Passo 2: Il Modello Avanzato**. Spiegazione teorica delle CNN profonde e del concetto di *Transfer Learning* (perché riutilizzare i pesi di ResNet50/EfficientNet accelera e migliora l'apprendimento).
5. **Addestramento e Hyperparameter Tuning**
   - Architettura della pipeline di training.
   - Scelta della *Loss Function* (es. CrossEntropy) e dell'*Optimizer* (es. Adam).
   - Gestione dell'overfitting (uso di Dropout ed Early Stopping).
6. **Risultati Sperimentali e Valutazione**
   - Analisi delle curve di Apprendimento (Loss e Accuracy per epoche di training e validation).
   - Valutazione sul Test Set tramite metriche avanzate (Accuracy globale, Precision, Recall, F1-Score per ogni classe).
   - Analisi della Matrice di Confusione: capire *dove* il modello sbaglia (es. confonde due malattie simili?).
7. **Conclusioni**
   - Riepilogo del lavoro e potenziale per implementazioni future (es. integrazione in app mobile o sistemi IoT come Arduino).


================================================================================
================================================================================

# PROGETTO ALTERNATIVO: Skincare & Color Palette AI [Opzione NON scelta]

Se decidi di puntare su questo progetto estremamente creativo e moderno, ecco tutto il materiale pronto da inviare al professore. Questo progetto unisce l'Apprendimento Supervisionato (per i problemi della pelle) e l'Apprendimento Non Supervisionato (per i colori), il che lo rende un progetto da lode assicurata.

## 1. Proposta Formale da Inviare al Professore (Skincare AI)

**Oggetto:** Proposta Progetto ML&DM - Analisi Facciale per Skincare e Armocromia tramite Deep Learning e Clustering

Gentile Professore,

Vorrei sottoporle una proposta di progetto per il corso di Machine Learning e Data Mining focalizzata sulla Computer Vision applicata al settore Beauty/Health-tech. L'obiettivo è creare un sistema in grado di analizzare immagini facciali per riconoscere problematiche della pelle e analizzare il sottotono per estrarre la palette stagionale (es. Winter, Summer), suggerendo di conseguenza routine di skincare e makeup.

**Dataset:** Costruirò una pipeline unendo dataset di immagini facciali e dermatologiche pubblici (es. *Ocular Disease Intelligent Recognition* o *Acne04* riadattati, combinati con dataset di facial skin tone).

**Metodologia e Tecnologie (Python, PyTorch, Scikit-Learn):**
1. **Riconoscimento Problematiche (Apprendimento Supervisionato)**: Sviluppo di un modello di Image Classification basato su Reti Neurali Convoluzionali (CNN) e Transfer Learning (es. ResNet/EfficientNet) per identificare problematiche cutanee (es. acne, rughe, discromie).
2. **Analisi Armocromia (Apprendimento Non-Supervisionato)**: Utilizzo di algoritmi di Clustering (K-Means / Gaussian Mixture Models) sui pixel dell'incarnato per raggruppare i colori dominanti, estrarre il sottotono e mappare il risultato su una delle palette stagionali.
3. **Valutazione**: Confronto del modello CNN tramite Accuracy e Matrice di Confusione, e valutazione del modello di Clustering tramite Silhouette Score o Elbow Method.

Il tutto sarà sviluppato in ambiente Google Colab. Unire tecniche supervisionate e non supervisionate mi permetterà di toccare più punti fondamentali del Suo corso. 
Resto in attesa di un Suo riscontro.

Cordiali saluti.

---

## 2. Scaletta della Relazione (Skincare AI)

1. **Introduzione e Definizione del Problema**
   - Contesto applicativo: Intelligenza Artificiale nella cosmetica personalizzata e dermatologia preventiva.
   - I due obiettivi: Rilevamento patologie cutanee (Supervised) ed Estrazione Palette Colori (Unsupervised).
2. **Esplorazione e Analisi dei Dati (EDA)**
   - Raccolta dei dati: presentazione dei dataset scelti per volti e texture della pelle.
   - Pre-processing delle immagini: estrazione automatica del volto (Face Detection) per ignorare lo sfondo.
3. **Pipeline 1: Apprendimento Supervisionato (Riconoscimento Pelle)**
   - Addestramento di una CNN Baseline per il rilevamento di patologie.
   - Implementazione di un modello avanzato con Transfer Learning (ResNet50).
   - Metriche di validazione e Matrice di Confusione.
4. **Pipeline 2: Apprendimento Non-Supervisionato (Clustering Armocromia)**
   - Trasformazione dello spazio colore (es. da RGB a HSV o LAB per isolare la luminosità).
   - Applicazione dell'algoritmo K-Means per trovare i colori dominanti della pelle.
   - Mappatura matematica dei centroidi trovati verso la "Stagione" corrispondente.
5. **Sistema di Suggerimento (Regole di Business)**
   - Unione dei due output (Condizione Pelle + Palette Stagione) per generare un piano Skincare/Makeup consigliato tramite regole logiche.
6. **Risultati Sperimentali e Limiti**
   - Performance combinate del sistema. 
   - Limiti etici e tecnici (es. importanza dell'illuminazione nella foto per non falsare l'armocromia).
7. **Conclusioni**


================================================================================
================================================================================

# CONFRONTO E VALUTAZIONE: QUALE PROGETTO SCEGLIERE? [Alla fine è stato scelto il progetto plant disease]

Questa sezione è progettata per aiutarti a prendere una decisione definitiva, valutando pragmaticamente i rischi, il tuo effort umano e le metriche di successo.

| Metrica di Valutazione | Progetto 1: Plant Disease | Progetto 2: Skincare & Armocromia |
| :--- | :--- | :--- |
| **Effort Umano (Codice)** | **Basso/Medio**: Il codice è lineare. C'è una sola pipeline (classificare l'immagine). | **Molto Alto**: Due algoritmi totalmente diversi da far dialogare e pipeline di pre-processing del viso. |
| **Effort Umano (Relazione)**| **Medio**: I grafici sono standard, facili da descrivere. | **Alto**: Spiegare due modelli matematici diversi (CNN e K-Means) raddoppia i capitoli da scrivere. |
| **Ricerca Dati (Dataset)** | **Facile**: Il dataset "PlantVillage" è già pulito, diviso per cartelle e bilanciato. | **Molto Difficile**: Bisogna unire più dataset sporchi (uno per l'acne, uno per i toni della pelle). |
| **Tempo Stimato (Totale)** | **3-5 giorni**: Altissima probabilità di finire comodamente in tempo senza stress notturni. | **7-10 giorni**: Rischio elevatissimo di andare oltre i 5 giorni previsti a causa dei dati frammentati. |
| **Impatto sul Professore** | **Voto 10-11**: Solido, accademico, dimostra un'ottima padronanza del Deep Learning. | **Voto 11+ (Lode garantita)**: Ambiziosissimo e innovativo, copre l'apprendimento Supervisionato e Non-Supervisionato insieme. |
| **Rischi Tecnici** | **Bassi**: Difficilmente il modello non riuscirà a convergere (imparare). | **Alti**: L'algoritmo dei colori può fallire miseramente se le foto di test hanno ombre o luci gialle/blu. |

## Sintesi e Consiglio Finale

> [!TIP]
> **Scegli le Piante se:** Vuoi dormire tranquillo. Avremo un progetto elegante, sicuro e da voto altissimo. Riuscirai a concentrarti sulla stesura di un'ottima relazione PowerPoint senza l'ansia che il codice si rompa il giorno prima della consegna.

> [!WARNING]
> **Scegli la Skincare se:** Vuoi presentare un progetto "spaccamascella" e sei disposto a lavorare intensamente (anche la sera) nei prossimi 5 giorni. Devi accettare il rischio ingegneristico di dover eventualmente ridimensionare le funzionalità all'ultimo minuto se non troviamo un dataset sufficientemente pulito per addestrare i due algoritmi in tempo.

---

## User Review Required

> [!IMPORTANT]
> Alla luce di questa tabella comparativa, scegli quale delle due proposte (Piante o Skincare) inviare al Professore.

## Open Questions

> [!CAUTION]
> Quando il prof approva una delle due, fammi sapere quale ha scelto e partiamo immediatamente a scrivere il codice su Colab!
