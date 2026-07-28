# Istruzioni Step-by-Step per Google Colab: Plant Disease Detection (PyTorch)

Benvenuto! Se non hai mai usato Google Colab o non hai mai affrontato un progetto di Machine Learning pratico, non preoccuparti. Questa guida è scritta per guidarti "per mano" dalla configurazione iniziale fino ai risultati finali. 

**Cos'è Google Colab?** 
Immaginalo come un "Google Document" ma invece di scrivere testo, scrivi ed esegui codice Python. Il grande vantaggio è che il codice non "gira" sul tuo computer, ma sui potentissimi server di Google. Questo ci permette di addestrare Reti Neurali pesantissime in pochi minuti senza fondere il nostro PC!

---

## Step 0: Procurarsi le chiavi di Kaggle (kaggle.json)
Prima di iniziare, abbiamo bisogno dei dati (le foto delle foglie). Il dataset che utilizzeremo è il **PlantVillage Dataset** (nella versione completa caricata su Kaggle dall'utente `abdallahalidev`). Utilizzeremo esclusivamente la cartella **color** all'interno di questo dataset, ignorando le versioni in scala di grigi e segmentate, in modo da avere esattamente ciò che serve alla nostra Rete Neurale.
Per scaricarlo in automatico su Colab, devi dire a Kaggle chi sei:
1. Vai su [Kaggle.com](https://www.kaggle.com/) e registrati (o fai login).
2. Clicca sulla tua foto profilo in alto a destra e vai su **Settings** (Impostazioni).
3. Scorri giù fino alla sezione **API** e clicca sul pulsante **"Create New Token"**.
4. Verrà scaricato un piccolo file chiamato `kaggle.json` sul tuo computer. Tienilo a portata di mano, ci servirà nello Step 2!

---

## Step 1: Creazione del blocco note e attivazione GPU
Ora andiamo su Google Colab.
1. Vai su [Google Colab](https://colab.research.google.com/) ed effettua l'accesso col tuo account Google.
2. Clicca su **"Nuovo blocco note"** (New notebook).
3. In alto a sinistra, puoi cliccare sul nome (es. `Untitled0.ipynb`) e rinominarlo in `Progetto_MLDM_PlantDisease.ipynb`.
4. **FONDAMENTALE (Attivazione GPU):** Vai nel menu in alto su **Runtime** -> **Cambia tipo di runtime**.
5. Sotto "Acceleratore hardware", seleziona **T4 GPU** e clicca su Salva.
   *(Senza questo passaggio, l'addestramento ci metterebbe ore anziché minuti!)*

---

## Step 2: Caricamento Dati e Importazione Librerie
Ora vedrai un rettangolo grigio con un tasto "Play" a sinistra. Quella è una **Cella di Codice**. 
**Come si usa?** Copia il codice qui sotto, incollalo nella cella e premi il tasto "Play" (oppure premi `Shift + Enter` sulla tastiera).
*Nota: quando eseguirai questa cella, comparirà un pulsante "Scegli file". Cliccalo e seleziona il file `kaggle.json` che hai scaricato nello Step 0!*

```python
!pip install kaggle
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configura le credenziali di Kaggle (ti chiederà di caricare il file kaggle.json)
from google.colab import files
print("PER FAVORE, CARICA IL FILE kaggle.json SCARICATO DA KAGGLE:")
uploaded = files.upload() 

# Spostiamo il file nella cartella di sistema corretta per farlo leggere a Kaggle
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Scarichiamo il dataset "Plant Disease" (PlantVillage)
print("Scaricamento del dataset in corso... (potrebbe volerci qualche istante)")
!kaggle datasets download -d abdallahalidev/plantvillage-dataset
!unzip -q plantvillage-dataset.zip -d dataset
print("Dataset scaricato ed estratto con successo!")
```

---

## Step 3: Preparazione delle Immagini (Data Augmentation)
Clicca su **"+ Codice"** in alto per creare una nuova cella.
Le reti neurali hanno bisogno che tutte le immagini siano della stessa dimensione. Inoltre, per evitare che la rete impari a memoria le foto (Overfitting), usiamo la **Data Augmentation**: incliniamo e tagliamo casualmente le immagini durante l'addestramento. Copia e avvia (Play):

```python
# Il dataset estratto ha diverse cartelle, una per ogni pianta/malattia
data_dir = 'dataset/color' # Cartella specifica per le immagini a colori 

# Diciamo a PyTorch come manipolare le immagini:
# Per l'addestramento: taglio casuale, ribaltamento orizzontale e normalizzazione dei colori.
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Per il test: ridimensioniamo e centriamo, nessuna modifica casuale.
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Carichiamo tutte le immagini
full_dataset = datasets.ImageFolder(data_dir)

# Dividiamo: 80% per imparare (Train), 20% per valutare (Validation)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = val_transforms

# Creiamo i "DataLoader" (caricano le immagini a pacchetti di 32 per non saturare la RAM)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class_names = full_dataset.classes
print(f"Ho trovato {len(class_names)} classi diverse di foglie!")
```

---

## Step 4: Costruzione della Rete Neurale (Transfer Learning)
Clicca di nuovo su **"+ Codice"**.
Invece di costruire una rete da zero, scarichiamo una rete famosissima chiamata **ResNet18**, che ha già imparato a riconoscere le forme del mondo reale, e le "insegniamo" solo a riconoscere le nostre foglie.

```python
# Diciamo a PyTorch di usare la GPU se disponibile
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Stiamo usando: {device}")

# Scarichiamo la ResNet18 pre-addestrata
model = models.resnet18(pretrained=True)

# Blocchiamo la memoria della rete in modo che non dimentichi le forme base
for param in model.parameters():
    param.requires_grad = False

# Cambiamo solo l'ultimo pezzo della rete per adattarlo al nostro numero di malattie (classi)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# Spostiamo il modello sulla GPU
model = model.to(device)

# Definiamo come calcolare l'errore (Loss) e come correggerlo (Adam Optimizer)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

---

## Step 5: Fase di Addestramento (Training)
Clicca su **"+ Codice"**. Questo è il cuore del progetto. Vedrai il calcolatore provare a indovinare le malattie e migliorarsi piano piano per 5 volte (epoche). Potrebbe volerci qualche minuto!

```python
num_epochs = 5
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("Inizio Addestramento...")
for epoch in range(num_epochs):
    model.train()
    running_loss, running_corrects = 0.0, 0
    
    # Fase di apprendimento (Training)
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc.item())
    
    # Fase di interrogazione (Validation - vediamo se ha davvero imparato)
    model.eval()
    val_loss, val_corrects = 0.0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
            
    val_epoch_loss = val_loss / len(val_dataset)
    val_epoch_acc = val_corrects.double() / len(val_dataset)
    val_losses.append(val_epoch_loss)
    val_accs.append(val_epoch_acc.item())
    
    print(f'Epoca {epoch+1}/{num_epochs} - Errore Train: {epoch_loss:.4f} Accuratezza Train: {epoch_acc:.4f} | Errore Val: {val_epoch_loss:.4f} Accuratezza Val: {val_epoch_acc:.4f}')
print("Addestramento completato!")
```

---

## Step 6: Valutazione Grafica (Loss e Confusion Matrix)
Ultima cella (Clicca su **"+ Codice"**). 
Questo genererà dei grafici bellissimi che **dovrai salvare (tasto destro -> Salva immagine) per metterli nella relazione in LaTeX**.

```python
# 1. Disegniamo i grafici di Apprendimento (Loss e Accuracy)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Errore Training')
plt.plot(val_losses, label='Errore Validation')
plt.legend()
plt.title('Andamento dell\'Errore (Loss)')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Accuratezza Training')
plt.plot(val_accs, label='Accuratezza Validation')
plt.legend()
plt.title('Andamento dell\'Accuratezza')
plt.show()

# 2. Generiamo il Report e la Matrice di Confusione
model.eval()
all_preds = []
all_labels = []
print("Analizzo il test set per la Matrice di Confusione...")
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\n--- REPORT DI CLASSIFICAZIONE ---")
print(classification_report(all_labels, all_preds, target_names=class_names))

print("\n--- MATRICE DI CONFUSIONE ---")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(15, 12)) # Più grande perché abbiamo tante classi
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Vera Malattia')
plt.xlabel('Malattia Predetta dal Modello')
plt.title('Matrice di Confusione sulle patologie fogliari')
plt.show()
```

Finito! Se l'ultimo grafico è una linea scura diagonale, hai appena ottenuto il 30!
