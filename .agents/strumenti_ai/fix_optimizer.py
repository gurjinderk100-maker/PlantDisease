import json

NB_PATH = "Progetto_Plant_Disease_Detection.ipynb"

def fix_notebook():
    with open(NB_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell['cell_type'] != 'code':
            continue
            
        source = "".join(cell['source'])
        
        # Fix optimizer_resnet accessing .fc on DataParallel
        target_str = "optimizer_resnet = optim.Adam(resnet_model.fc.parameters(), lr=0.001)"
        if target_str in source:
            fix_str = """# Fix per nn.DataParallel: dobbiamo accedere al modulo interno se presente
m_resnet = resnet_model.module if isinstance(resnet_model, nn.DataParallel) else resnet_model
optimizer_resnet = optim.Adam(m_resnet.fc.parameters(), lr=0.001)"""
            source = source.replace(target_str, fix_str)
            print("[OK] Bug dell'optimizer corretto!")

        # Update the cell source
        lines = source.split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []

    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        print("[OK] Notebook salvato!")

if __name__ == "__main__":
    fix_notebook()
