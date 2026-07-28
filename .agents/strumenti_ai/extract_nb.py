import json
import sys

def extract_notebook(ipynb_path, output_path):
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    with open(output_path, 'w', encoding='utf-8') as out:
        for i, cell in enumerate(nb.get('cells', [])):
            cell_type = cell.get('cell_type', '')
            out.write(f"\n--- CELL {i} ({cell_type}) ---\n")
            source = cell.get('source', [])
            if isinstance(source, list):
                out.write("".join(source))
            else:
                out.write(source)
            out.write("\n")

if __name__ == "__main__":
    extract_notebook("noteobok(5).ipynb", "extracted_final_nb.txt")
