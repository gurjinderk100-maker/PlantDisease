from PIL import Image
import os
import shutil
import matplotlib.pyplot as plt

# 1. Gradio interface
shutil.copy("schermata sito web .png", "relazione_latex/figure/gradio_interface.png")

# 2. Grad-CAM image
img = Image.open("image (1).webp").convert("RGB")
img.save("relazione_latex/figure/gradcam_heatmap.png")

# 3. Original leaf
shutil.copy("original_leaf.JPG", "relazione_latex/figure/original_leaf.jpg")

# 4. Create dataset mosaic
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
dirs = [
    "esempi immagini dataset random/tomato__late_blight",
    "esempi immagini dataset random/strawberry__leaf_scorch",
    "esempi immagini dataset random/corn_maize_northen_leaf_blight"
]

for i, d in enumerate(dirs):
    files = os.listdir(d)
    img_path = os.path.join(d, files[0])
    img = Image.open(img_path)
    axes[i].imshow(img)
    axes[i].axis('off')
    title = d.split('/')[-1].replace('__', ' - ').replace('_', ' ').title()
    axes[i].set_title(title, fontsize=10)

plt.tight_layout()
plt.savefig("relazione_latex/figure/dataset_samples.png", dpi=150, bbox_inches='tight')
print("Images processed and saved to figure/")
