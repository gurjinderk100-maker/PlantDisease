# Glossario dei Termini Tecnici

Questo documento raccoglie in modo semplice e accessibile tutti i principali termini tecnici utilizzati nella relazione del progetto.

## Modelli e Architetture

- **Baseline (Modello di base)**: Un modello semplice creato per stabilire un punto di partenza. Le prestazioni di questo modello "base" servono come metro di paragone per capire se modelli più complessi portano un reale vantaggio. In questo progetto, la baseline è una rete neurale creata da zero.
- **Convolutional Neural Network (CNN)**: Un tipo speciale di rete neurale artificiale ispirata alla corteccia visiva degli animali, progettata appositamente per elaborare immagini. Funziona "scansionando" l'immagine alla ricerca di forme e colori (linee, cerchi, macchie).
- **ResNet-50 (Residual Network 50)**: Un modello di rete neurale molto profondo (con 50 strati sovrapposti). Ha rivoluzionato il mondo della Computer Vision introducendo delle "scorciatoie" (connessioni residuali) che permettono di creare modelli enormi senza che si "dimentichino" le informazioni durante l'apprendimento.
- **Support Vector Machine (SVM)**: Un potente algoritmo matematico di classificazione. Immagina i dati sparsi su un foglio: l'SVM cerca di tracciare la linea più netta e spessa possibile per separare i dati appartenenti a categorie diverse.

## Metodologie e Tecniche

- **Computer Vision**: Il campo dell'intelligenza artificiale che insegna ai computer a "vedere" e comprendere le immagini digitali o i video, estrapolandone informazioni utili (ad esempio, capire se c'è una malattia in una foglia).
- **Transfer Learning (Apprendimento Trasferito)**: Una tecnica in cui si prende un modello già addestrato su un compito molto generico (come riconoscere animali o oggetti comuni, ad esempio usando ImageNet) e lo si riadatta per un compito specifico (come riconoscere malattie delle piante). Permette di ottenere risultati eccellenti con meno dati e in minor tempo.
- **EDA (Exploratory Data Analysis)**: L'analisi esplorativa dei dati. È la fase iniziale in cui si studiano i dati a disposizione (in questo caso, quante foto abbiamo, di quali piante, quante foto sono sane e quante malate) per capire se ci sono problemi da risolvere prima di iniziare l'addestramento.
- **K-Means Clustering**: Un algoritmo di intelligenza artificiale classico che divide automaticamente un gruppo di dati in "K" gruppi (cluster) basandosi sulle loro somiglianze. In questo progetto è stato usato per vedere se l'IA riusciva a raggruppare da sola le malattie simili.

## Strumenti Visivi e Dataset

- **ImageNet**: Un gigantesco database contenente milioni di immagini suddivise in migliaia di categorie diverse (cani, gatti, automobili, ecc.). Viene utilizzato per addestrare modelli molto grandi in modo che imparino a riconoscere il mondo visivo in generale.
- **Grad-CAM (Gradient-weighted Class Activation Mapping)**: Una tecnica visiva che produce una "mappa di calore" sopra un'immagine. Le zone colorate in rosso o giallo indicano le parti dell'immagine che la rete neurale ha guardato con più attenzione per prendere la sua decisione. È molto utile per verificare che l'IA non stia barando (es. guardando lo sfondo invece della foglia).
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Uno strumento di visualizzazione statistica. Dato che i modelli di intelligenza artificiale pensano in centinaia o migliaia di dimensioni (impossibili da disegnare o immaginare), il t-SNE "schiaccia" queste dimensioni in sole 2 o 3, permettendoci di disegnare un grafico e vedere visivamente se il modello ha raggruppato bene le malattie o se le sta confondendo.
