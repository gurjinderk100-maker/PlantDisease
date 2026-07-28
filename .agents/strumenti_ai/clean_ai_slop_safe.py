import json

NB_PATH = "Progetto_Plant_Disease_Detection.ipynb"

# Mappa delle sostituzioni (SOLO TESTO, no regex)
replacements = {
    "✅ ": "",
    "🚀 ": "",
    "🧠 ": "",
    "🎉 ": "",
    "🏆 ": "",
    "🌿 ": "",
    "🟢 ": "",
    "👉 ": "",
    "⚠️ ": "",
    "🖼️ ": "",
    "💾 ": "",
    "📊 ": "",
    "📄 ": "",
    "IL TOCCO DA ESPERTO: ": "Nota architetturale: ",
    "Riduce qualsiasi mappa di feature (es. 128x56x56) a 128x1x1": "Riduzione della dimensionalità delle feature maps a 1x1 tramite GAP",
    "Ora l'input è sempre 128, indipendentemente dalla grandezza dell'immagine!": "L'input ai layer fully-connected è ora invariante rispetto alla dimensione dell'immagine originaria.",
    "Cervello Baseline CNN caricato con successo!": "Modello Baseline CNN caricato con successo.",
    "Cervello ResNet50 caricato con successo!": "Modello ResNet50 caricato con successo.",
    "TRUCCO MAGICO: Creiamo delle \\\"history\\\" finte così la cella 7.5 dei grafici": "Inizializzazione array storici fittizi per compatibilità grafica in assenza di training dinamico:",
    "spostato in alto per evitare l'errore!": "calcolato preventivamente",
    "spostato in alto per evitare errori": "calcolato preventivamente",
    "Training completato in": "Addestramento completato in",
    "Migliore accuratezza Val:": "Migliore accuratezza Validation:",
    "CLICCA SU QUESTO LINK PER APRIRE TENSORBOARD:\\n": "URL TensorBoard generato:\\n",
    "ATTENZIONE: Quando apri il link, per sicurezza ti chiederà un 'Endpoint IP'.": "ATTENZIONE: Autenticazione Endpoint IP richiesta.",
    "Copia e incolla esattamente questo numero qui sotto:": "Inserire il seguente indirizzo IP per procedere:",
    "In quest'ultima sezione uniamo la potenza estrattiva della Rete Neurale con le tecniche classiche di Machine Learning e visualizzazione (PCA/t-SNE e Support Vector Machines).": "In questa sezione si applica un approccio ibrido, estraendo features tramite Deep Learning per alimentare modelli di Machine Learning classico (SVM) e tecniche di riduzione della dimensionalità (t-SNE).",
    "Per rendere la fase di caricamento immagini molto più intuitiva, utilizziamo Gradio. Questa libreria genererà un'interfaccia utente web integrata direttamente qui nel notebook, dove potrai trascinare le tue immagini (Drag & Drop) e vedere in tempo reale sia la classificazione che l'analisi Grad-CAM.": "Implementazione di un'interfaccia interattiva tramite Gradio per il test manuale dei campioni, con visualizzazione real-time delle predizioni e delle mappe di attivazione Grad-CAM.",
    "Carica la foto di una foglia malata per scoprire la patologia e vedere dove la rete neurale si è concentrata.": "Caricamento campione per test di classificazione e visualizzazione focus dell'algoritmo (Grad-CAM).",
    "In base alle richieste del progetto, salviamo i grafici della Loss e dell'Accuracy per poterli includere nella relazione finale.": "Generazione e salvataggio dei grafici di Loss e Accuracy per l'analisi dei risultati di addestramento.",
    "Generazione scatter plot t-SNE...": "Generazione visualizzazione t-SNE in corso...",
    "Riduciamo le dimensioni del test set da N-D a 2D per poterle plottare": "Riduzione dimensionale da 512D a 2D per visualizzazione scatter plot",
    "Creiamo una palette di colori per le 38 classi": "Inizializzazione palette colori per visualizzazione classi multiclasse",
    "Spostiamo la legenda fuori dal grafico per evitare sovrapposizioni": "Formattazione legenda",
    "SALVATAGGIO IN ALTA DEFINIZIONE (DPI 300) PER LA RELAZIONE": "Salvataggio immagine output a 300 DPI",
    "Cervello Baseline caricato!": "Modello Baseline caricato.",
    "Cervello ResNet50 caricato!": "Modello ResNet50 caricato."
}

def clean_ai_slop():
    with open(NB_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if not cell.get('source'):
            continue
            
        # Sostituiamo stringa per stringa senza fare join e split pericolosi
        new_source = []
        for line in cell['source']:
            new_line = line
            for old, new in replacements.items():
                new_line = new_line.replace(old, new)
            new_source.append(new_line)
            
        cell['source'] = new_source

    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Notebook pulito e professionalizzato con successo.")

if __name__ == "__main__":
    clean_ai_slop()
