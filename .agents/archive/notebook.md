# Progetto Machine Learning & Data Mining: Plant Disease Detection

Questo notebook sviluppa una pipeline completa di **Deep Learning** per la classificazione automatica delle patologie fogliari a partire da immagini digitali, utilizzando il dataset *PlantVillage*.

### Fasi del Notebook:
1. **Setup e Ingestione Dati**: Verifica ambiente GPU e download automatico tramite API Kaggle.
2. **Analisi Esplorativa dei Dati (EDA)**: Ispezione del dataset, bilanciamento delle classi e visualizzazione campioni.
3. **Data Augmentation & Preprocessing**: Definizione delle trasformazioni visive e suddivisione *Train/Validation/Test*.
4. **Modello Baseline (Custom CNN)**: Architettura di riferimento definita da zero con PyTorch.
5. **Modello Avanzato (Transfer Learning)**: Fine-tuning di architetture pre-addestrate (es. ResNet50).
6. **Valutazione e Metriche**: Matrice di confusione, Accuracy, Precision, Recall ed F1-Score.

## 1. Setup dell'Ambiente e Importazione Librerie
!pip install -q kaggle grad-cam tensorboard gradio

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from itertools import cycle
import requests
from io import BytesIO

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize

import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import gradio as gr

# Configurazione Device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Device attivo: GPU CUDA ({torch.cuda.get_device_name(0)})")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device attivo: Apple Silicon MPS")
else:
    device = torch.device("cpu")
    print("Device attivo: CPU")

# Fissiamo i seed per la riproducibilità
import random
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)
print('Seed impostato per la riproducibilità.')

# Se esegui in locale, decommentare le righe sottostanti per il download via API
# if not os.path.exists('/kaggle/input'):
#     os.system("kaggle datasets download -d emmarex/plantdisease")
#     os.system("unzip -q plantdisease.zip -d dataset")

## 2.5 Analisi Esplorativa dei Dati (EDA)
In questa fase analizziamo la composizione del dataset scaricato, verificando il numero di classi, il bilanciamento e visualizzando alcuni campioni rappresentativi per assicurarci della bontà dei dati prima del preprocessing.

print("Ricerca del dataset in corso...")
start_time = time.time()

possible_paths = [
    '/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color',
    '/kaggle/input/plantvillage-dataset/color',
    '/kaggle/input/plantdisease/dataset/color',
    'dataset/color'
]
import glob
data_dir = next((p for p in possible_paths if os.path.exists(p)), None)
if data_dir is None:
    found_paths = glob.glob('/kaggle/input/**/color', recursive=True)
    if found_paths:
        data_dir = found_paths[0]

if data_dir:
    print(f"Directory trovata: {data_dir}")
    
    class_counts = {}
    total_images = 0
    classes = []
    
    with os.scandir(data_dir) as entries:
        for entry in entries:
            if entry.is_dir():
                classes.append(entry.name)
                
    for c in classes:
        class_path = os.path.join(data_dir, c)
        VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        with os.scandir(class_path) as it:
            count = sum(1 for entry in it if entry.is_file() and os.path.splitext(entry.name)[1].lower() in VALID_EXTENSIONS)
        class_counts[c] = count
        total_images += count
        
    print(f"Scansione completata in {time.time() - start_time:.2f} secondi.")
    print(f"Totale Immagini: {total_images} | Totale Classi: {len(classes)}")
    
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    x_labels = [x[0] for x in sorted_classes]
    y_values = [x[1] for x in sorted_classes]
    
    plt.figure(figsize=(15, 6))
    sns.barplot(x=x_labels, y=y_values, hue=x_labels, palette="viridis", legend=False)
    plt.xticks(rotation=90, fontsize=8)
    plt.title("Distribuzione Immagini per Classe Fitosanitaria")
    plt.ylabel("Conteggio")
    plt.tight_layout()
    plt.show()
    plt.close('all')
else:
    raise RuntimeError("Dataset non trovato! Assicurati di aver scaricato i dati.")

## 3. Data Augmentation e Preparazione DataLoader
from torchvision import transforms
import os

if 'data_dir' not in locals() or data_dir is None:
    data_dir = 'dataset/color'

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if os.path.exists(data_dir):
    print("Trasformazioni definite correttamente.")
    print(f"Dataset pronto per il caricamento da: {data_dir}\n")
else:
    print(f"Attenzione: Il percorso {data_dir} non è valido.")


## 4. Suddivisione del Dataset (Train, Validation, Test)
import platform
_num_workers = 4 if platform.system() == 'Linux' else 0

from torch.utils.data import DataLoader, random_split
from torchvision import datasets

if os.path.exists(data_dir):
    full_dataset = datasets.ImageFolder(data_dir)

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    targets = full_dataset.targets
    
    from sklearn.model_selection import train_test_split
    
    train_idx, temp_idx, _, temp_targets = train_test_split(
        range(len(targets)), targets, test_size=0.3, stratify=targets, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_targets, random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

    class DatasetWrapper(Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
        def __len__(self):
            return len(self.subset)

    train_dataset = DatasetWrapper(train_dataset, transform=train_transforms)
    val_dataset = DatasetWrapper(val_dataset, transform=val_test_transforms)
    test_dataset = DatasetWrapper(test_dataset, transform=val_test_transforms)

    _batch_size = 128
    _prefetch = 2 if _num_workers > 0 else None
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=_batch_size, shuffle=True, num_workers=_num_workers, prefetch_factor=_prefetch, pin_memory=torch.cuda.is_available()),
        'val': DataLoader(val_dataset, batch_size=_batch_size, shuffle=False, num_workers=_num_workers, prefetch_factor=_prefetch, pin_memory=torch.cuda.is_available()),
        'test': DataLoader(test_dataset, batch_size=_batch_size, shuffle=False, num_workers=_num_workers, prefetch_factor=_prefetch, pin_memory=torch.cuda.is_available())
    }

    print(f"Trovate {len(full_dataset.classes)} classi.")
    print(f"Dimensioni dataset: Train={train_size}, Val={val_size}, Test={test_size}")
else:
    print("Il dataset non è presente, impossibile creare i dataloaders.")


## 5. Definizione del Modello Baseline (Custom CNN)
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1)) 
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

num_classes_detected = len(full_dataset.classes)
baseline_model = CustomCNN(num_classes=num_classes_detected)
if torch.cuda.device_count() > 1:
    print(f"Uso {torch.cuda.device_count()} GPU per il modello Baseline")
    baseline_model = nn.DataParallel(baseline_model)
baseline_model = baseline_model.to(device)
print(f"Modello Baseline inizializzato per {num_classes_detected} classi.")


## 6. Definizione del Modello Avanzato (Transfer Learning - ResNet50)
def build_resnet50_model(num_classes, freeze_backbone=True):
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    if freeze_backbone:
        for param in resnet.parameters():
            param.requires_grad = False
            
    in_features = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return resnet

num_classes_detected = len(full_dataset.classes)
resnet_model = build_resnet50_model(num_classes=num_classes_detected)
if torch.cuda.device_count() > 1:
    print(f"Uso {torch.cuda.device_count()} GPU per la ResNet50")
    resnet_model = nn.DataParallel(resnet_model)
resnet_model = resnet_model.to(device)
print(f"Modello ResNet50 configurato in Transfer Learning per {num_classes_detected} classi.")


## 7. Pipeline di Training e Validazione
import time
import copy
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=10, device='cuda', save_path='model.pth', writer_name='runs/esperimento'):
    since = time.time()
    
    writer = SummaryWriter(writer_name)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    val_key = 'val' if 'val' in dataloaders else 'test'

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 15)

        for phase in ['train', val_key]:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()}", leave=False)

            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
                writer.add_scalar('Loss/Train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/Train', epoch_acc, epoch)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                writer.add_scalar('Loss/Validation', epoch_loss, epoch)
                writer.add_scalar('Accuracy/Validation', epoch_acc, epoch)
                
                if scheduler is not None:
                    scheduler.step(epoch_loss)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), save_path)

    time_elapsed = time.time() - since
    print(f'\nTraining completato in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Migliore accuratezza Val: {best_acc:.4f}')

    writer.close()
    model.load_state_dict(best_model_wts)
    return model, history


## 7.1 Visualizzazione TensorBoard
import subprocess
import time

!pkill -f tensorboard
!pkill -f localtunnel

subprocess.Popen(['tensorboard', '--logdir', 'runs/', '--host', '0.0.0.0', '--port', '6006'])
time.sleep(3)

print("Generazione del link esterno per TensorBoard...")
tunnel = subprocess.Popen(['npx', 'localtunnel', '--port', '6006'], stdout=subprocess.PIPE)
time.sleep(3)
url = tunnel.stdout.readline().decode('utf-8').strip()

print("\n" + "="*65)
print(f"Link per aprire TensorBoard:\n{url}")
print("="*65 + "\n")
print("Attenzione: Quando apri il link, verrà richiesto un 'Endpoint IP'.")
print("Copia e incolla questo numero:")
!curl -s ipv4.icanhazip.com

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch

train_indices = train_dataset.subset.indices
train_labels = [full_dataset.targets[i] for i in train_indices]
class_counts = np.bincount(train_labels)
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights = class_weights.to(device)

print("--- INIZIO TRAINING MODELLO BASELINE ---")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer_baseline = optim.Adam(baseline_model.parameters(), lr=0.001)

scheduler_baseline = ReduceLROnPlateau(optimizer_baseline, mode='min', factor=0.1, patience=2)

baseline_model, history_baseline = train_model(
    baseline_model, dataloaders, criterion, optimizer_baseline, 
    num_epochs=10, device=device, save_path='baseline_cnn.pth', scheduler=scheduler_baseline, writer_name='runs/baseline_cnn'
)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Memoria GPU liberata.")

print("\n--- INIZIO TRAINING TRANSFER LEARNING (ResNet50) ---")

criterion = nn.CrossEntropyLoss(weight=class_weights)

m_resnet = resnet_model.module if isinstance(resnet_model, nn.DataParallel) else resnet_model
optimizer_resnet = optim.Adam(m_resnet.fc.parameters(), lr=0.001)

scheduler_resnet = ReduceLROnPlateau(optimizer_resnet, mode='min', factor=0.1, patience=2)

resnet_model, history_resnet = train_model(
    resnet_model, dataloaders, criterion, optimizer_resnet, 
    num_epochs=10, device=device, save_path='resnet50_best.pth', scheduler=scheduler_resnet, writer_name='runs/resnet50'
)


## 7.5 Salvataggio e Visualizzazione delle Curve di Addestramento
def plot_and_save_history(history, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_title(f'{title} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    filename = f'{title.replace(" ", "_").lower()}_history.png'
    plt.savefig(filename)
    print(f"Grafico salvato: {filename}")
    plt.show()
    plt.close('all')

if 'history_baseline' in locals():
    plot_and_save_history(history_baseline, title="Baseline CNN")

if 'history_resnet' in locals():
    plot_and_save_history(history_resnet, title="ResNet50 Transfer Learning")


## 8. Valutazione Finale e Matrici di Confusione
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, dataloaders, class_names, model_name='model'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders['test'], desc=f"Valutazione {model_name}"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    model_slug = model_name.replace(" ", "_").lower()
    
    report_txt = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print("\n--- REPORT DI CLASSIFICAZIONE ---")
    print(report_txt)
    
    txt_filename = f'classification_report_{model_slug}.txt'
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write(report_txt)
    print(f"Report testuale salvato in: {txt_filename}")
    
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    csv_report_filename = f'classification_report_{model_slug}.csv'
    df_report.to_csv(csv_report_filename)
    print(f"Report in tabella salvato in: {csv_report_filename}")
    
    print("\n--- MATRICE DI CONFUSIONE ---")
    cm = confusion_matrix(all_labels, all_preds)
    
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    csv_cm_filename = f'confusion_matrix_{model_slug}.csv'
    df_cm.to_csv(csv_cm_filename)
    print(f"Dati grezzi della matrice salvati in: {csv_cm_filename}")
    
    plt.figure(figsize=(24, 22)) 
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.ylabel('Classe Reale', fontsize=14)
    plt.xlabel('Classe Predetta', fontsize=14)
    plt.title(f'Matrice di Confusione - {model_name}', fontsize=18)
    
    img_filename = f'confusion_matrix_{model_slug}.png'
    plt.tight_layout()
    plt.savefig(img_filename, dpi=300, bbox_inches='tight')
    print(f"Immagine della matrice salvata in: {img_filename}")
    plt.show()
    plt.close('all')

if 'test' in dataloaders:
    print("\n=======================")
    print("VALUTAZIONE BASELINE CNN")
    print("=======================")
    if 'baseline_model' in locals():
        evaluate_model(baseline_model, dataloaders, full_dataset.classes, model_name="Baseline CNN")

    print("\n=======================")
    print("VALUTAZIONE RESNET50")
    print("=======================")
    if 'resnet_model' in locals():
        evaluate_model(resnet_model, dataloaders, full_dataset.classes, model_name="ResNet50")


## 9.1 Interfaccia Web Interattiva (Gradio)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import gradio as gr
import torch.nn.functional as F

def make_predict_fn(model, class_names, transform, device):
    def predict_gradio(img):
        if img is None:
            return None, None
            
        m = model.module if isinstance(model, nn.DataParallel) else model
        m.eval()
        
        pil_img = Image.fromarray(img).convert('RGB')
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = m(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
        
        top3_prob, top3_catid = torch.topk(probabilities, 3)
        confidences = {class_names[top3_catid[i]]: float(top3_prob[i]) for i in range(3)}
        
        if hasattr(m, 'layer4'):
            target_layers = [m.layer4[-1]]
        elif hasattr(m, 'features'):
            target_layers = [m.features[-1]]
        else:
            target_layers = []
            
        if target_layers:
            cam = GradCAM(model=m, target_layers=target_layers)
            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(torch.argmax(probabilities).item())])[0, :]
            img_resized = (np.array(pil_img.resize((224, 224))) / 255.0).astype(np.float32)
            cam_image = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)
        else:
            cam_image = None
            
        return confidences, cam_image
    return predict_gradio

if 'resnet_model' in locals():
    print("Avvio dell'interfaccia web Gradio...")
    interface = gr.Interface(
        fn=make_predict_fn(resnet_model, full_dataset.classes, val_test_transforms, device),
        inputs=gr.Image(label="Carica un'immagine della foglia"),
        outputs=[
            gr.Label(num_top_classes=3, label="Previsioni"),
            gr.Image(label="Grad-CAM (Aree di attenzione)")
        ],
        title="Plant Disease Detector",
        description="Carica la foto di una foglia per analizzare la patologia e le feature considerate dal modello."
    )
    interface.launch(share=True, prevent_thread_lock=True)


## 10. Analisi Avanzate Ibride (Machine Learning Classico + Deep Learning)
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def extract_features(model, dataloader):
    model.eval()
    features = []
    labels_list = []
    
    m = model.module if isinstance(model, nn.DataParallel) else model
    feature_extractor = torch.nn.Sequential(*list(m.children())[:-1]).to(device)
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = feature_extractor(inputs)
            outputs = outputs.view(outputs.size(0), -1) 
            features.append(outputs.cpu().numpy())
            labels_list.append(labels.numpy())
            
    return np.vstack(features), np.concatenate(labels_list)

if 'resnet_model' in locals() and 'test' in dataloaders:
    print("\n--- MODELLO IBRIDO: CNN Feature Extractor + Support Vector Machine ---")
    
    from torch.utils.data import Subset
    import random
    
    svm_train_data = dataloaders['train'].dataset
    subset_indices = random.sample(range(len(svm_train_data)), int(0.2 * len(svm_train_data)))
    train_subset = Subset(svm_train_data, subset_indices)
    subset_dataloader = DataLoader(train_subset, batch_size=32, shuffle=False)
    
    print("Estrazione features dal Train Set...")
    X_train_feat, y_train_feat = extract_features(resnet_model, subset_dataloader)
    
    print("Estrazione features dal Test Set...")
    X_test_feat, y_test_feat = extract_features(resnet_model, dataloaders['test'])
    
    from sklearn.svm import SVC
    print("Addestramento SVM in corso...")
    svm_clf = SVC(kernel='linear', C=1.0, probability=True, random_state=42, class_weight='balanced')
    svm_clf.fit(X_train_feat, y_train_feat)
    
    svm_preds = svm_clf.predict(X_test_feat)
    svm_acc = accuracy_score(y_test_feat, svm_preds)
    print(f"Accuracy del modello Ibrido (ResNet50 + SVM): {svm_acc*100:.2f}%")


## -- VISUALIZZAZIONE DELLO SPAZIO LATENTE CON t-SNE --
if 'X_test_feat' in locals():
    print("\n Generazione scatter plot t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_test_feat)
    
    plt.figure(figsize=(14, 10)) 
    palette = sns.color_palette("husl", len(full_dataset.classes))
    for i, c in enumerate(full_dataset.classes):
        idxs = (y_test_feat == i)
        plt.scatter(X_tsne[idxs, 0], X_tsne[idxs, 1], color=palette[i], label=c, alpha=0.6, s=15)
        
    plt.title("Visualizzazione t-SNE delle Features Estratte", fontsize=16)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', markerscale=2, fontsize=8, ncol=2)
    
    plt.savefig('tsne_features_plot.png', dpi=300, bbox_inches='tight')
    print("Grafico t-SNE salvato come tsne_features_plot.png")
    
    plt.show()
    plt.close('all')


## -- CURVE ROC E AUC PER MULTICLASSE --
if 'svm_clf' in locals() and 'X_test_feat' in locals():
    print("\n Calcolo delle Curve ROC e AUC per il Modello Ibrido...")
    y_test_bin = label_binarize(y_test_feat, classes=range(len(full_dataset.classes)))
    n_classes = y_test_bin.shape[1]
    
    y_score = svm_clf.predict_proba(X_test_feat)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(5), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC {0} (AUC = {1:0.3f})'.format(full_dataset.classes[i][:15], roc_auc[i]))
                 
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    macro_auc = np.mean([roc_auc[i] for i in range(n_classes)])
    
    plt.plot(fpr["micro"], tpr["micro"], color='black', lw=3, linestyle=':',
             label=f'Micro-Average ROC (AUC = {roc_auc["micro"]:.3f})')
    plt.plot([], [], ' ', label=f'Macro-Average AUC = {macro_auc:.3f}')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) - Top 5 Classi')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves_multiclass.png')
    print("Curve ROC salvate come roc_curves_multiclass.png")
    plt.show()
    plt.close('all')


## 11. Apprendimento Non Supervisionato: Clustering con K-Means
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import os

if 'X_test_feat' in locals():
    print("\n--- CLUSTERING NON SUPERVISIONATO (K-MEANS) ---")
    
    # Impostiamo il numero di cluster (K) pari al numero di classi reali
    num_clusters = len(full_dataset.classes)
    print(f"Esecuzione di K-Means con K={num_clusters} sulle feature estratte dalla ResNet50...")
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_test_feat)
    
    sil_score = silhouette_score(X_test_feat, cluster_labels)
    print(f"Silhouette Score: {sil_score:.4f}")
    
    # Salvataggio CSV dei risultati del clustering
    out_dati = 'dati'
    os.makedirs(out_dati, exist_ok=True)
    df_kmeans = pd.DataFrame({
        'Classe_Reale_ID': y_test_feat,
        'Classe_Reale_Nome': [full_dataset.classes[i] for i in y_test_feat],
        'Cluster_KMeans_Assegnato': cluster_labels
    })
    csv_kmeans_path = os.path.join(out_dati, 'kmeans_cluster_assignments.csv')
    df_kmeans.to_csv(csv_kmeans_path, index=False)
    print(f"Salvato file CSV: {csv_kmeans_path}")
    
    # Visualizzazione dei cluster sul t-SNE
    if 'X_tsne' in locals():
        plt.figure(figsize=(14, 10))
        
        # Generiamo una palette di colori
        palette = sns.color_palette("husl", num_clusters)
        
        # Plot dei punti suddivisi per cluster per avere la legenda
        for k in range(num_clusters):
            idxs = (cluster_labels == k)
            plt.scatter(X_tsne[idxs, 0], X_tsne[idxs, 1], color=palette[k], label=f'Cluster {k}', alpha=0.7, s=20)
        
        plt.title(f"Clustering K-Means (Silhouette Score: {sil_score:.3f})", fontsize=16)
        
        # Aggiunta della legenda fuori dal grafico
        plt.legend(title="Cluster Assegnato", bbox_to_anchor=(1.01, 1), loc='upper left', markerscale=1.5, fontsize=8, ncol=2)
        
        plt.tight_layout()
        plot_kmeans_path = os.path.join(out_dati, 'kmeans_clusters_plot.png')
        plt.savefig(plot_kmeans_path, dpi=300, bbox_inches='tight')
        print(f"Grafico salvato: {plot_kmeans_path}")
        
        plt.show()
        plt.close('all')


## Salvataggio finale completo
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from itertools import cycle
import torch

output_dir = 'dati'
os.makedirs(output_dir, exist_ok=True)
print(f"Cartella di destinazione creata/verificata: '{output_dir}/'\n")

if 'class_counts' in locals():
    if isinstance(class_counts, dict):
        df_classes = pd.DataFrame(list(class_counts.items()), columns=['Classe', 'ConteggioImmagini'])
    else:
        names = full_dataset.classes if 'full_dataset' in locals() else range(len(class_counts))
        df_classes = pd.DataFrame({'Classe': names, 'ConteggioImmagini': class_counts})
    df_classes.to_csv(os.path.join(output_dir, 'class_distribution.csv'), index=False)
    print(f"Salvato: {output_dir}/class_distribution.csv")

if 'class_weights' in locals():
    weights_cpu = class_weights.cpu().numpy() if isinstance(class_weights, torch.Tensor) else class_weights
    names = full_dataset.classes if 'full_dataset' in locals() else range(len(weights_cpu))
    df_weights = pd.DataFrame({'Classe': names, 'Peso_Loss': weights_cpu})
    df_weights.to_csv(os.path.join(output_dir, 'class_weights.csv'), index=False)
    print(f"Salvato: {output_dir}/class_weights.csv")

if 'history_baseline' in locals() and 'history_resnet' in locals():
    summary_data = {
        'Modello': ['Baseline Custom CNN', 'ResNet50 Transfer Learning'],
        'Parametri Totali': ['~136,000', '~23,500,000'],
        'Best Val Accuracy (%)': [f"{max(history_baseline['val_acc'])*100:.2f}", f"{max(history_resnet['val_acc'])*100:.2f}"],
        'Best Val Loss': [f"{min(history_baseline['val_loss']):.4f}", f"{min(history_resnet['val_loss']):.4f}"]
    }
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(output_dir, 'summary_model_comparison.csv'), index=False)
    print(f"Salvato: {output_dir}/summary_model_comparison.csv")

def export_history(history, title):
    slug = title.replace(" ", "_").lower()
    
    df_hist = pd.DataFrame(history)
    df_hist.index.name = 'Epoch'
    df_hist.index += 1
    csv_path = os.path.join(output_dir, f'{slug}_history_data.csv')
    df_hist.to_csv(csv_path)
    print(f"Salvato: {csv_path}")

if 'history_baseline' in locals():
    export_history(history_baseline, "Baseline CNN")

if 'history_resnet' in locals():
    export_history(history_resnet, "ResNet50 Transfer Learning")

if 'resnet_model' in locals() and 'test' in dataloaders:
    resnet_model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            outputs = resnet_model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    class_names = full_dataset.classes
    
    cm = confusion_matrix(all_labels, all_preds)
    
    cm_temp = cm.copy()
    np.fill_diagonal(cm_temp, 0)
    errors = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm_temp[i, j] > 0:
                errors.append({
                    'Classe_Reale': class_names[i],
                    'Classe_Predetta_Errata': class_names[j],
                    'Numero_Errori': cm_temp[i, j]
                })
    df_err = pd.DataFrame(errors).sort_values(by='Numero_Errori', ascending=False).head(10)
    df_err.to_csv(os.path.join(output_dir, 'top_10_misclassifications.csv'), index=False)
    print(f"Salvato: {output_dir}/top_10_misclassifications.csv")

if 'svm_clf' in locals() and 'X_test_feat' in locals():
    y_test_bin = label_binarize(y_test_feat, classes=range(len(full_dataset.classes)))
    n_classes = y_test_bin.shape[1]
    y_score = svm_clf.predict_proba(X_test_feat)
    
    fpr, tpr, roc_auc = {}, {}, {}
    auc_data = []
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        auc_data.append({'Classe': full_dataset.classes[i], 'AUC_Score': roc_auc[i]})
        
    df_auc = pd.DataFrame(auc_data)
    df_auc.to_csv(os.path.join(output_dir, 'roc_auc_scores.csv'), index=False)
    print(f"Salvato: {output_dir}/roc_auc_scores.csv")

print("\nProcesso di salvataggio completato nella cartella 'dati/'")
