# Concetto 2: Valutazione dei Modelli e Metriche
*Riferimento slide del corso: Evaluation-aprile-2021-web.pdf*

## Overfitting: Cos'è e come l'abbiamo evitato?
"L'overfitting è quando il modello impara a memoria i dati di training ma non riesce a generalizzare su dati nuovi (come uno studente che impara a memoria gli esercizi ma non sa fare l'esame). 
Nel nostro grafico (curve di apprendimento), capiamo che NON c'è overfitting perché la curva di *Validation Loss* scende e rimane stabile, senza risalire improvvisamente mentre la *Training Loss* scende a zero.
Lo abbiamo evitato grazie alla **Data Augmentation**: abbiamo ruotato, specchiato e zoomato artificialmente le immagini di addestramento, costringendo la rete a capire la malattia a prescindere dall'inclinazione della foglia."

## Precision, Recall e Matrice di Confusione
(Fondamentali per l'esame)
- **Accuracy**: Quante volte ci ho azzeccato in totale. (Es. 95% delle foglie classificate giuste). Ma in problemi reali non basta!
- **Precision (Precisione)**: "Di tutte le foglie che il modello ha etichettato come *Malate*, quante lo erano davvero?"
- **Recall (Sensibilità)**: "Di tutte le foglie *realmente Malate* nel dataset, quante il modello è riuscito a trovarne?"

Nel caso delle patologie agricole, avere una **Recall alta** è fondamentale: è meglio avere un "falso allarme" (una foglia sana etichettata come malata) piuttosto che mancare una foglia infetta che poi distruggerà l'intero raccolto!

- **Matrice di Confusione**: È quella tabella a scacchiera che abbiamo stampato con Seaborn. Ci fa vedere esattamente *quali classi si confondono tra loro*. I numeri alti sulla diagonale principale sono le predizioni corrette. I numeri fuori dalla diagonale sono gli errori.
