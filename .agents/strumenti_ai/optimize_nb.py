import json

NB_PATH = "Progetto_Plant_Disease_Detection.ipynb"

def optimize_notebook():
    with open(NB_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell['cell_type'] != 'code':
            continue
            
        source = "".join(cell['source'])
        
        # 1. DataLoader Optimization
        if "dataloaders = {" in source and "batch_size=32" in source:
            # Replace num_workers logic
            source = source.replace("_num_workers = 2 if platform.system() == 'Linux' else 0", 
                                    "_num_workers = 4 if platform.system() == 'Linux' else 0")
            
            # Replace dataloaders dict
            new_dataloaders = """
    # OTTIMIZZAZIONE GPU/CPU BOTTLENECK
    _batch_size = 128
    _prefetch = 2 if _num_workers > 0 else None
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=_batch_size, shuffle=True, num_workers=_num_workers, prefetch_factor=_prefetch, pin_memory=torch.cuda.is_available()),
        'val': DataLoader(val_dataset, batch_size=_batch_size, shuffle=False, num_workers=_num_workers, prefetch_factor=_prefetch, pin_memory=torch.cuda.is_available()),
        'test': DataLoader(test_dataset, batch_size=_batch_size, shuffle=False, num_workers=_num_workers, prefetch_factor=_prefetch, pin_memory=torch.cuda.is_available())
    }
"""
            # Find the start of the dataloaders dict
            start_idx = source.find("    dataloaders = {")
            if start_idx != -1:
                # Find the end of the dataloaders dict
                end_idx = source.find("    }", start_idx) + 5
                source = source[:start_idx] + new_dataloaders + source[end_idx:]
                print("[OK] DataLoader ottimizzato!")

        # 2. Baseline Model DataParallel
        if "baseline_model = CustomCNN(num_classes=num_classes_detected).to(device)" in source:
            new_baseline = """baseline_model = CustomCNN(num_classes=num_classes_detected)
if torch.cuda.device_count() > 1:
    print(f"🚀 Uso {torch.cuda.device_count()} GPU per il modello Baseline!")
    baseline_model = nn.DataParallel(baseline_model)
baseline_model = baseline_model.to(device)"""
            source = source.replace("baseline_model = CustomCNN(num_classes=num_classes_detected).to(device)", new_baseline)
            print("[OK] Baseline Model ottimizzato per Multi-GPU!")
            
        # 3. ResNet50 DataParallel
        if "resnet_model = build_resnet50_model(num_classes=num_classes_detected).to(device)" in source:
            new_resnet = """resnet_model = build_resnet50_model(num_classes=num_classes_detected)
if torch.cuda.device_count() > 1:
    print(f"🚀 Uso {torch.cuda.device_count()} GPU per la ResNet50!")
    resnet_model = nn.DataParallel(resnet_model)
resnet_model = resnet_model.to(device)"""
            source = source.replace("resnet_model = build_resnet50_model(num_classes=num_classes_detected).to(device)", new_resnet)
            print("[OK] ResNet50 ottimizzata per Multi-GPU!")

        # Update the cell source, keeping it as a list of lines for valid JSON formatting
        # We split by \n and add \n back except for the last line
        lines = source.split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []

    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        print("[OK] Notebook salvato con successo!")

if __name__ == "__main__":
    optimize_notebook()
