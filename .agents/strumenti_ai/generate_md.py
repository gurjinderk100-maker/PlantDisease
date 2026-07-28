import json

def genera_md():
    with open('Progetto_Plant_Disease_Detection.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    with open('Codice_Notebook_Completo.md', 'w', encoding='utf-8') as f_out:
        f_out.write("# Codice Completo del Notebook (Senza AI Slop)\n\n")
        f_out.write("Copia e incolla questi blocchi nelle celle di Kaggle.\n\n")
        
        for i, cell in enumerate(nb.get('cells', [])):
            tipo = "Testo (Markdown)" if cell['cell_type'] == 'markdown' else "Codice (Python)"
            f_out.write(f"## Cella {i+1} - {tipo}\n")
            
            if cell['cell_type'] == 'code':
                f_out.write("```python\n")
            
            # Scrive il contenuto
            source = "".join(cell.get('source', []))
            f_out.write(source)
            
            if cell['cell_type'] == 'code':
                f_out.write("\n```\n")
                
            f_out.write("\n---\n\n")

if __name__ == "__main__":
    genera_md()
