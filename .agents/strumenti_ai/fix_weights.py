import json
import numpy as np

NB_PATH = "Progetto_Plant_Disease_Detection.ipynb"

def fix_notebook_weights():
    with open(NB_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell['cell_type'] != 'code':
            continue
            
        source = "".join(cell['source'])
        
        # Cerca la definizione dell'addestramento baseline
        if "criterion = nn.CrossEntropyLoss(weight=class_weights)" in source and "# Calcolo Class Weights" not in source:
            
            # Codice da iniettare
            weights_calc_code = """
# Calcolo Class Weights per gestire lo sbilanciamento (Spostato qui in alto!)
import numpy as np
train_indices = train_dataset.subset.indices
train_labels = [full_dataset.targets[i] for i in train_indices]
class_counts = np.bincount(train_labels)
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
"""
            # Sostituiamo la riga singola con il blocco completo
            source = source.replace("criterion = nn.CrossEntropyLoss(weight=class_weights)", weights_calc_code.strip())
            
            # Poi andiamo a rimuovere (o commentare) la vecchia definizione che era nella sezione ResNet
            # per evitare di ricalcolarli inutilmente, anche se non farebbe male, ma puliamo
            old_weights_code = """# Calcolo Class Weights per gestire lo sbilanciamento
train_indices = train_dataset.subset.indices
train_labels = [full_dataset.targets[i] for i in train_indices]
class_counts = np.bincount(train_labels)
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights = class_weights.to(device)"""
            if old_weights_code in source:
                source = source.replace(old_weights_code, "# (Class weights già calcolati in alto)")
                
            print("[OK] Bug del class_weights corretto (Spacchettamento in alto)!")

        # Update the cell source
        lines = source.split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []

    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        print("[OK] Notebook salvato!")

if __name__ == "__main__":
    fix_notebook_weights()
