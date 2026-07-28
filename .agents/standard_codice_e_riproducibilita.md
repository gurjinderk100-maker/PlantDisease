# Standard di Codice, Riproducibilità e Best Practices

Questo file stabilisce le norme tecniche che l'IA e gli sviluppatori devono seguire nello sviluppo del codice Python e dei Notebook Jupyter.

---

## 1. Garanzia di Riproducibilità
Ogni notebook o script Python deve contenere all'inizio il fissaggio esplicito dei seed casuali per garantire la riproducibilità totale degli esperimenti:

```python
import os
import random
import numpy as np
import torch

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
```

---

## 2. Gestione Flessibile dell'Hardware (CPU / GPU / MPS)
Il codice deve sempre verificare in modo dinamico la presenza di acceleratori hardware per funzionare indistintamente su Google Colab, Apple Silicon Mac o CPU locali:

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device utilizzato per l'addestramento: {device}")
```

---

## 3. Struttura Standard dei Notebook Jupyter

1. **Cella 1: Setup & Importazioni** (import librerie, gestione seed, configurazione device).
2. **Cella 2: Ingestione Dati** (Download automatico tramite Kaggle API o script Python).
3. **Cella 3: EDA (Exploratory Data Analysis)** (Ispezione dimensioni, valori nulli, sbilanciamento classi, campioni visivi).
4. **Cella 4: Data Augmentation & Loaders** (Split 80% Train, 10% Val, 10% Test con `random_split`, trasformazioni PyTorch).
5. **Cella 5: Architettura Baseline** (Modello personalizzato da zero).
6. **Cella 6: Architettura Avanzata** (Transfer Learning / ResNet / VGG).
7. **Cella 7: Loop di Addestramento & Tracciamento** (Loss e Accuracy per epoca, salvataggio miglior modello).
8. **Cella 8: Valutazione Finale & Metriche** (Classification report, Matrice di confusione su Test Set, visualizzazione errori).

---

## 4. Gestione delle Credenziali Kaggle in Colab
Per i notebook trascritti per Colab, utilizzare il blocco standard non bloccante per il caricamento del file `kaggle.json`:

```python
import os

try:
    from google.colab import files
    if not os.path.exists('/root/.kaggle/kaggle.json'):
        print("Caricare il file kaggle.json:")
        uploaded = files.upload()
        os.system("mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json")
except Exception as e:
    print(f"Esecuzione in ambiente locale o credenziali già presenti: {e}")
```
