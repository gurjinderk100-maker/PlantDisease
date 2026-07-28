# Linee Guida Esame, Consegna e Criteri di Valutazione (MLDM UniBS)

Questo file contiene le informazioni chiave sulla modalità d'esame, le regole di consegna dell'elaborato pratico ed i criteri per massimizzare il punteggio (puntando a 30/30 e Lode).

---

## 1. Struttura dell'Esame (Totale 30 Punti)

L'esame si compone di due prove indipendenti (senza propedeuticità) valide per l'intero anno accademico:

1. **Prova Scritta (20 Punti su 30)**:
   - Durata: 2 ore.
   - Tipologia: Domande a risposta aperta sugli argomenti teorici del corso (Prof. Serina e Prof. Gerevini).

2. **Prova Pratica / Elaborato di Progetto (10 Punti su 30)**:
   - Svolta singolarmente o in gruppi (1-3 studenti).
   - Prevede la realizzazione di una pipeline di Machine Learning / Data Mining e la stesura di una **relazione scritta**.
   - Discussione dell'elaborato durante un **colloquio orale** con i docenti.

---

## 2. Requisiti di Consegna dell'Elaborato Pratico

All'atto della consegna (tramite form/portale del corso), è necessario allegare:
1. **Relazione Scritta (PDF)**: Documento completo contenente abstract, motivazioni, descrizione dati, architetture utilizzate, risultati sperimentali e conclusioni.
2. **Presentazione (PDF)**: Slide sintetiche per il colloquio orale (tipicamente 10-15 slide).
3. **Archivio ZIP**: Contenente tutti gli script Python (`.py`), i notebook Jupyter (`.ipynb`) ed i dataset (o script di scaricamento automatico da Kaggle/UCI).

---

## 3. Criteri per il Voto Massimo (30/30)

Dall'analisi dei progetti degli anni passati con valutazione massima (voto 10-11+ / 10), i progetti di maggior successo presentano le seguenti caratteristiche:

- **Combinazione Supervisionato + Non Supervisionato**: Integrare sia modelli di classificazione/regressione (es. Reti Neurali / Random Forest / SVM) sia tecniche di clustering (es. K-Means / DBSCAN per l'analisi dei cluster delle immagini o delle feature).
- **Confronto tra Baseline e Modello Avanzato**: Includere sempre un modello semplice di riferimento (es. Custom CNN o modello lineare) e confrontarlo rigorosamente con modelli avanzati (es. Transfer Learning con ResNet50 o Ensembling).
- **Rigore Sperimentale e Grafici**: Presentare metriche complete (Accuracy, Precision, Recall, F1-Score, Loss Curves, Matrice di Confusione, curve ROC) e visualizzazioni chiare dei risultati.
- **Formattazione Pulita e Impersonale**: Relazione scritta con registro accademico impersonale (forma passiva/implicita).
