## AUD: Are You Dangerous?

## Descrizione del progetto

**AUD** (Are You Dangerous) è un progetto che si occupa del riconoscimento e della classificazione delle azioni umane utilizzando dati scheletrici. Attraverso l'uso di tecniche avanzate di deep learning, il sistema è in grado di analizzare pose e movimenti per identificare le azioni svolte da una persoma all'interno di un video.

## Struttura del repository

Il repository è organizzato nei seguenti file:

- **`parseSkeleton.py`** – Effettua il parsing dei dati grezzi dello scheletro in un formato strutturato e utilizzabile.
- **`generateSequence.py`** – Genera sequenze di dati scheletrici per essere utilizzate come input dal modello.
- **`augmentSkeleton.py`** – Permette di generare file aggiuntivi per il dataset, usando data augmentation, in modo da rendere piu robbusto il modello. 
- **`CNN_GRUModel.py`** – Definizione dell'architettura del modello basato su CNN-GRU per il riconoscimento delle azioni.
- **`dataLoader.py`** – Gestisce il caricamento e la preparazione dei dati scheletrici per l'addestramento e la valutazione.
- **`trainEnhanced.py`** – Script per l'addestramento del modello con configurazioni ottimizzate.
- **`valutazioneEnhanced.py`** – Valutazione del modello addestrato tramite metriche di classificazione.
- **`demo.py`** – Script per testare il modello su nuovi video e visualizzare i risultati con sovraimpressione delle azioni riconosciute.
  
## Installazione e configurazione

### 1. Clonare il repository

```bash
git clone https://github.com/chrizzo3/AUD.git
cd AUD 
```

### 2. Installare le dipendeze 
Assicurarsi di aver intallato Python dopo di che eseguire:
```bash
pip install -r requirements.txt
```
### 3. Preparare i dati
Nella cartella dataset sono presenti tutti gli scheletri necessari all'addestramento del modello. Utilizzare il file parseSkeleton.py per trasformare estrarre i file dalle cartelle .zip e trasformarli in file json. 
```bash
py parseSkeleton.py
```
I file json hanno la sequente struttura
filename: il nome del file 
action_label: l'identificativo dell'azione contenuta nel file
action_name: il nome dell'azione
frames:[
    frame_index: il numero del frame attuale
    num bodie: il numero di scheletri rilevato in quel frame
    bodies[
        body_id: identificativo del corpo
        joints:[
            "3D_position": {
                                "x": 
                                "y": 
                                "z": 
                            }
                            "orientation": {
                                "w": 
                                "x": 
                                "y": 
                                "z": 
                            }
                            "tracking_state": 
        ]  per tutti e 25 i joints
    ]  per ogni scheletro contenuto nel frame
]  per ogni frame del video

### 5. Generare le sequenze
Eseguire 
```bash
py generateSequences.py
```
Trasforma i file json in sequenze di lunghezza 30 salvate in array npy.

### 6. Generare i dati sintetici 
Eseguire:
```bash
py augmentSkeleton.py
```
Genera sequenze di scheletri sintetiche generate tramite l'aggiunta di rotazione, rumore e ridimensionamento delle sequenze gia prodotte in precedenza. Inoltre il codice divide tutte le sequenze in train_file, val_file e test_file con proporzione 70-30-30.

### 7. Addestramento del modello
Per avviare l'addestramento del modello eseguire:
```bash
py trainEnhanced.py
```

###8 8. Valutazione del modello
E' possibile valutare il modello tramite:
```bash
py valutazioneEnhanced.py
```
Se il modello non rispetta le prestazioni desiderate è possibile modificare le caratteristiche del modello in [CNN_GRuModel.py](CNN_GRUModel.py), cambiare le caratteristiche anche nell'inializzazione dle modello in [trainEnhanced.py](trainEnhanced.py) e [valutazioneEnhanced.py](valutazioneEnhanced.py)

### 9. Demo
Per provare il modello è possibile eseguire:
```bash
py demo.py
```
sostuire il path del video in ingresso con il video che si desidera usare
