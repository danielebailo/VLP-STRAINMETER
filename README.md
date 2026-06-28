# VLP-STRAINMETER — Classificatore CNN per Waveform Discrimination

**Classificatore supervisionato basato su CNN 1D per distinguere waveform VLP di tipo 1 da tipo 2.**

## Panoramica

Questo progetto addestra un classificatore neurale per discriminare automaticamente due classi di waveform sismiche (VLP — Very Long Period) da dati strainmeter:

- **Classe 0 (Type1)** — waveform di tipo 1
- **Classe 1 (Type2)** — waveform di tipo 2

Il classificatore apprende direttamente dalle etichette di classe e fornisce:
- Previsioni con probabilità per ogni classe
- Metriche di valutazione (accuracy, AUC, ROC curve, confusion matrix)
- Visualizzazione interattiva dei risultati su dati reali

## Pipeline

```
1. Carica dati         → Type1 e Type2 da file .txt
2. Normalizza          → Global mean/std su tutti i dati
3. Split               → 80% train / 10% val / 10% test
4. Addestra CNN        → 4 blocchi convoluzionali + fully connected
5. Valuta              → Accuracy, AUC, confusion matrix, ROC
6. Predici su reali    → Classificazione su VLPs-2019_low-pass.dat
7. Visualizza          → Grafici interattivi Plotly
```

## Architettura del Modello

```
Input: (batch, 1, 128)
  │
  ├── Conv1d(1→32, k=7, s=2, p=3) → BatchNorm → ReLU → Dropout1d
  ├── Conv1d(32→64, k=5, s=2, p=2) → BatchNorm → ReLU → Dropout1d
  ├── Conv1d(64→128, k=5, s=2, p=2) → BatchNorm → ReLU → Dropout1d
  ├── Conv1d(128→256, k=3, s=2, p=1) → BatchNorm → ReLU → Dropout1d
  ├── AdaptiveAvgPool1d(1)
  │
  ├── Flatten → Linear(256→64) → ReLU → Dropout
  └── Linear(64→2) → Softmax → [P(Type1), P(Type2)]
```

## Struttura del Progetto

```
VLP-STRAINMETER/
├── config/
│   └── config.yaml           # Configurazione
├── data/
│   ├── processed/            # Dataset di training
│   │   ├── Waveforms_type1_extended_128samples.txt
│   │   └── Waveforms_type2_extended_128samples.txt
│   └── measured/
│       └── VLPs-2019_low-pass.dat   # Dati reali da classificare
├── src/
│   ├── classifier.py         # Modello, training, valutazione, predizione (PRINCIPALE)
│   ├── config.py             # Parser configurazione
│   ├── data.py               # Loader dati (autoencoder v1.0)
│   ├── models.py             # Autoencoder (v1.0, deprecato)
│   ├── train.py              # Training autoencoder (v1.0, deprecato)
│   ├── discriminator.py      # Discriminazione autoencoder (v1.0, deprecato)
│   └── plotting.py           # Plotting (v1.0)
├── main_classifier.py        # CLI entrypoint per il classificatore
├── notebooks/
│   ├── classifier.ipynb      # Notebook classificatore (PRINCIPALE)
│   └── strainmeter_workflow.ipynb  # Notebook autoencoder v1.0 (deprecato)
├── outputs/
│   └── cnn_classifier.pt     # Modello classificatore addestrato
├── CHANGELOG.md              # Storico delle evoluzioni
└── README.md                 # Questo file
```

## Come Eseguire

### Opzione 1: Jupyter Notebook (consigliato)

```bash
cd /Users/danielebailo/Hermes/workspace/projects/VLP-STRAINMETER
jupyter notebook notebooks/classifier.ipynb
```

Esegui le celle in ordine:
1. **Setup** — configurazione ambiente e device
2. **Carica dati** — loading e splitting train/val/test
3. **Addestra CNN** — training con grafico in tempo reale
4. **Valuta** — accuracy, AUC, confusion matrix, ROC
5. **Predici su reali** — classificazione delle prime 20 waveform
6. **Predici su tutte** — classificazione completa + grafico Plotly interattivo

### Opzione 2: CLI

```bash
python3 main_classifier.py \
  --type1 data/processed/Waveforms_type1_extended_128samples.txt \
  --type2 data/processed/Waveforms_type2_extended_128samples.txt \
  --epochs 50 \
  --lr 1e-3 \
  --device auto \
  --save outputs/cnn_classifier.pt
```

### Opzione 3: Import da codice

```python
from src.classifier import VLPClassifier, train_classifier, load_and_split, evaluate_model, predict

# Carica dati
train_loader, val_loader, test_loader = load_and_split(
    "data/processed/Waveforms_type1_extended_128samples.txt",
    "data/processed/Waveforms_type2_extended_128samples.txt",
    waveform_length=128,
)

# Addestra
model, history = train_classifier(train_loader, val_loader, epochs=50)

# Valuta
results = evaluate_model(model, test_loader)

# Predici su nuove waveforms
preds = predict(model, new_waveforms)
```

## Configurazione

Modifica `config/config.yaml` per:

| Parametro | Descrizione |
|-----------|-------------|
| `classifier.epochs` | Numero di epoche di training |
| `classifier.learning_rate` | Learning rate (Adam) |
| `classifier.dropout` | Dropout probability |
| `classifier.batch_size` | Batch size |
| `classifier.waveform_length` | Lunghezza waveform (default: 128) |
| `classifier.train_ratio` | Frazione per training (default: 0.8) |
| `classifier.val_ratio` | Frazione per validation (default: 0.1) |
| `classifier.test_ratio` | Frazione per test (default: 0.1) |
| `classifier.seed` | Random seed per riproducibilità |

## Dati

### Training Data

| File | Dimensione | Descrizione |
|------|-----------|-------------|
| `Waveforms_type1_extended_128samples.txt` | ~7.2MB | ~100k waveforms di tipo 1 |
| `Waveforms_type2_extended_128samples.txt` | ~5.3MB | ~75k waveforms di tipo 2 |

Formato: una waveform per riga, 128 campioni per riga.

### Dati Reali

| File | Dimensione | Descrizione |
|------|-----------|-------------|
| `VLPs-2019_low-pass.dat` | ~115MB | ~150k waveforms misurate (151 campioni ciascuna) |

Le waveform reali vengono center-cropped a 128 campioni prima della classificazione.

## Output

| File | Descrizione |
|------|-----------|
| `outputs/cnn_classifier.pt` | Modello addestrato (checkpoint) |
| `notebooks/outputs/confusion_matrix.png` | Matrice di confusione |
| `notebooks/outputs/roc_curve.png` | Curva ROC |

## Metriche

Il classificatore fornisce:

- **Accuracy** — percentuale di previsioni corrette
- **AUC-ROC** — area sotto la curva ROC (discriminazione tra classi)
- **Confusion Matrix** — TP, TN, FP, FN per classe
- **Probabilità** — P(Type1) e P(Type2) per ogni waveform
- **Confidenza** — |P(Type1) - P(Type2)| per ogni previsione

## Dipendenze

```
torch>=2.0
numpy
matplotlib
scikit-learn
pyyaml
plotly
jupyter
```

Installazione:
```bash
pip install -r requirements.txt
```

## Storico delle Versioni

Vedi [`CHANGELOG.md`](CHANGELOG.md) per la cronologia completa delle evoluzioni del progetto.

- **v2.0 (attuale)** — Classificatore CNN supervisionato
- **v1.0 (deprecato)** — Autoencoder 1D con discriminazione basata su errore di ricostruzione

## Note

- Il classificatore v2.0 è il percorso principale. I file v1.0 sono conservati per riferimento storico.
- Non usare per nuovo lavoro: `src/autoencoder.py`, `src/data_processor.py`, `src/waveform_discriminator.py`
- Per il classificatore, usa: `src/classifier.py`, `main_classifier.py`, `notebooks/classifier.ipynb`
