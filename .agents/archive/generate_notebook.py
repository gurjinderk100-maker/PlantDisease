import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells.append(nbf.v4.new_markdown_cell("# Inferenza e XAI (Grad-CAM & Gradio) con ResNet50\nQuesto notebook permette di caricare i pesi `resnet50_best.pth` senza rieseguire il training e assolve alle richieste per Grad-CAM e Gradio."))

nb.cells.append(nbf.v4.new_code_cell("""\
!pip install -q grad-cam gradio
import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import gradio as gr
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# Impostazioni Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in uso: {device}")

# 1. Definizione dell'architettura
m_resnet = models.resnet50(pretrained=False)
num_ftrs = m_resnet.fc.in_features
m_resnet.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 38)
)

# 2. Caricamento dei pesi (Assicurati che 'resnet50_best.pth' sia in /kaggle/working/ o nella cartella corrente)
model_path = 'resnet50_best.pth'
if os.path.exists(model_path):
    state_dict = torch.load(model_path, map_location=device)
    # Fix per i pesi salvati con DataParallel (rimuove il prefisso 'module.')
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    m_resnet.load_state_dict(new_state_dict)
    print("Pesi caricati con successo!")
else:
    print(f"ERRORE: File {model_path} non trovato. Caricalo su Kaggle/Colab e assicurati che il percorso sia corretto.")
    
m_resnet = m_resnet.to(device)
m_resnet.eval()
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# Nomi delle classi (PlantVillage 38 classi)
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

transform_inf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# Funzione Unificata per Predizione e Grad-CAM
def predict_gradio(img):
    if img is None:
        return "Nessuna immagine", None
        
    img_pil = Image.fromarray(img).convert('RGB')
    input_tensor = transform_inf(img_pil).unsqueeze(0).to(device)
    
    # Inferenza
    with torch.no_grad():
        output = m_resnet(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        conf, predicted = torch.max(probabilities, 0)
        predicted_class = class_names[predicted.item()]
        
    # Risultato formattato per Gradio
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    class_probs = {class_names[top5_catid[i]]: float(top5_prob[i]) for i in range(5)}
    
    # Grad-CAM
    target_layers = [m_resnet.layer4[-1]]
    try:
        cam = GradCAM(model=m_resnet, target_layers=target_layers)
        targets = [ClassifierOutputTarget(predicted.item())]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Normalizza immagine originale per sovrapposizione
        img_resized = cv2.resize(np.array(img_pil), (256, 256))
        img_normalized = np.float32(img_resized) / 255
        
        cam_image = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
        
        # SALVA LE IMMAGINI PER LA RELAZIONE AUTOMATICAMENTE
        plt.imsave("gradcam_heatmap_relazione.png", cam_image)
        plt.imsave("gradcam_original_relazione.png", img_resized)
        
    except Exception as e:
        print("Errore GradCAM:", e)
        cam_image = img
        
    return class_probs, cam_image
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# Avvio di Gradio Web UI
iface = gr.Interface(
    fn=predict_gradio,
    inputs=gr.Image(label="Carica un'immagine della foglia"),
    outputs=[
        gr.Label(num_top_classes=3, label="Previsioni"),
        gr.Image(label="Grad-CAM (Aree di attenzione)")
    ],
    title="🌿 Plant Disease Detector & Explainer",
    description="Carica la foto di una foglia malata per scoprire la patologia e vedere dove la rete neurale si è concentrata.",
    allow_flagging="never"
)

iface.launch(share=True)
"""))

with open('notebook_gradcam_gradio.ipynb', 'w') as f:
    nbf.write(nb, f)
