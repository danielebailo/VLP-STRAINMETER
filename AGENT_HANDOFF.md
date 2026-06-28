# STRAINMETER — Handoff per Classificatore v2.0

Questo file è il percorso più breve per un nuovo agent per comprendere rapidamente il progetto.

## Cos'è il progetto

Il repository implementa un **classificatore supervisionato basato su CNN 1D** per discriminare waveform VLP (Very Long Period) di tipo 1 da tipo 2.

**Fase attuale:** Classificatore CNN completo, addestrato e pronto per la predizione.

## Cosa fa

1. Carica due dataset di training (type1 e type2)
2. Normalizza globalmente (mean/std su tutti i dati)
3. Split in train (80%), val (10%), test (10%)
4. Addestra una CNN 1D profonda con 4 blocchi convoluzionali
5. Valuta con accuracy, AUC, confusion matrix, ROC curve
6. Predice su waveform reali (`VLPs-2019_low-pass.dat`)
7. Visualizza i risultati con grafici interattivi Plotly

## Entrypoint primari

- **Notebook:** `notebooks/classifier.ipynb` — workflow visuale completo
- **CLI:** `main_classifier.py` — esecuzione da terminale

Notebook modes (celle in ordine):
1. `Setup` — configurazione ambiente
2. `Carica dati` — loading e splitting
3. `Addestra CNN` — training con grafico in tempo reale
4. `Valuta` — metriche su test set
5. `Predici su reali` — classificazione prime 20 waveform
6. `Predici su tutte` — classificazione completa + Plotly

## Struttura cartelle

- `config/`: configurazione YAML
- `data/processed/`: waveforms di training (type1 e type2)
- `data/measured/`: file reale da classificare
- `notebooks/`: notebook Jupyter
- `outputs/`: artefatti generati (modello, grafici)
- `src/`: implementazione

## File chiave

- `src/classifier.py` — `VLPClassifier`, training, valutazione, predizione
- `main_classifier.py` — CLI entrypoint
- `notebooks/classifier.ipynb` — notebook completo
- `config/config.yaml` — configurazione

## Dati

Training:
- `data/processed/Waveforms_type1_extended_128samples.txt` (~7.2MB)
- `data/processed/Waveforms_type2_extended_128samples.txt` (~5.3MB)

Real:
- `data/measured/VLPs-2019_low-pass.dat` (~115MB)

Regole:
- ogni riga è una waveform
- training: 128 campioni
- real: 151 campioni → center-cropped a 128

## Architettura modello

```
Conv1d(1→32, k=7, s=2) → BN → ReLU → Dropout1d
Conv1d(32→64, k=5, s=2) → BN → ReLU → Dropout1d
Conv1d(64→128, k=5, s=2) → BN → ReLU → Dropout1d
Conv1d(128→256, k=3, s=2) → BN → ReLU → Dropout1d
AdaptiveAvgPool1d(1)
Flatten → Linear(256→64) → ReLU → Dropout → Linear(64→2)
```

## Training

- Loss: CrossEntropy
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau (patience=5)
- Epochs: 50 (configurabile)
- Batch size: 64
- Device: auto (MPS > CUDA > CPU)

## Output

- `outputs/cnn_classifier.pt` — modello addestrato
- `notebooks/outputs/confusion_matrix.png`
- `notebooks/outputs/roc_curve.png`

## Come modificare

### Cambiare dati di training
Modificare i percorsi in `main_classifier.py` o nel notebook

### Cambiare il modello
Modificare `src/classifier.py` → classe `VLPClassifier`

### Cambiare gli iperparametri
Modificare `config/config.yaml` o i parametri di `train_classifier()`

### Cambiare le metriche
Modificare `src/classifier.py` → funzione `evaluate_model()`

## Cosa NON usare

Questi sono file legacy della versione autoencoder (v1.0):
- `src/autoencoder.py`
- `src/data_processor.py`
- `src/waveform_discriminator.py`
- `src/models.py` (Conv1dAutoencoder)
- `src/train.py` (training autoencoder)
- `notebooks/strainmeter_workflow.ipynb`

Usare il classificatore invece:
- `src/classifier.py`
- `main_classifier.py`
- `notebooks/classifier.ipynb`

## Storico

Vedi [`CHANGELOG.md`](CHANGELOG.md) per la cronologia completa.

- **v2.0 (attuale)** — Classificatore CNN supervisionato
- **v1.0 (deprecato)** — Autoencoder 1D con discriminazione basata su errore di ricostruzione
