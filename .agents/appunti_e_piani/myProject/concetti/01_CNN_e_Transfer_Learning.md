# Concetto 1: Reti Neurali Convoluzionali (CNN) e Transfer Learning
*Riferimento slide del corso: 12-ANN.pdf e HML-Cap10-SlidesNN.pdf*

## Cosa dire all'orale in parole povere (se il prof ti chiede "Cos'è una CNN?")
"Professore, una CNN (Convolutional Neural Network) è un tipo speciale di rete neurale progettata apposta per analizzare dati che hanno una struttura a griglia, come le immagini. 
A differenza delle reti neurali classiche dove ogni neurone è connesso a tutti gli altri, la CNN usa un'operazione matematica chiamata **Convoluzione**. 
In pratica, passa un 'filtro' (una finestrella) sopra l'immagine per estrarre caratteristiche visive: prima i bordi, poi forme semplici, fino ad arrivare a riconoscere le macchie causate dalla malattia della foglia."

## Cosa è il Transfer Learning? (Domanda da 30 e lode)
"Invece di far imparare alla rete da zero cosa sia una linea o un cerchio (che richiederebbe milioni di foto e settimane di calcolo), abbiamo usato il **Transfer Learning**. Abbiamo preso un modello chiamato ResNet18, che Google/Microsoft avevano già addestrato su milioni di immagini (ImageNet). 
La rete sapeva già 'vedere'. Noi abbiamo semplicemente 'tagliato' l'ultimo pezzo della rete (il Fully Connected Layer finale) e lo abbiamo sostituito con un nuovo strato che impara a distinguere solo le nostre foglie malate. È come prendere un super-esperto generico di immagini e fargli fare un master in botanica."

## Perché usare PyTorch?
(Se ti chiede perché hai scelto PyTorch invece di Scikit-Learn per questo task):
"Mentre Scikit-Learn è perfetto per algoritmi tradizionali come SVM o Decision Trees, PyTorch è un framework nato apposta per il Deep Learning e il calcolo su GPU (Tensori). Ci ha permesso di importare facilmente architetture complesse come ResNet e velocizzare l'addestramento tramite Google Colab."
