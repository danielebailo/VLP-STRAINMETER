# Changelog — VLP-STRAINMETER

Tutte le modifiche rilevanti per il progetto.

---

## v2.0 — Classificatore CNN Supervisionato (attuale)

**Data:** Giugno 2026

### Cosa è cambiato

Il progetto è passato da un approccio **autoencoder-based** (reconstruction error) a un **classificatore supervisionato** con CNN 1D.

### Cosa fa v2.0

Addestra un classificatore neurale supervisionato per distinguere waveform VLP di tipo 1 da tipo 2:

1. **Carica** due dataset di training (type1 e type2)
2. **Normalizza** globalmente (mean/std su tutti i dati)
3. **Split** in train (80%), val (10%), test (10%)
4. **Addestra** una CNN 1D profonda con 4 blocchi convoluzionali
5. **Valuta** su test set con accuracy, AUC, confusion matrix, ROC curve
6. **Predice** su waveform reali (file `VLPs-2019_low-pass.dat`)
7. **Visualizza** i risultati con grafici interattivi Plotly

### Architettura del modello

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
  └── Linear(64→2) → Softmax
```

### File chiave

| File | Descrizione |
|------|-----------|
| `src/classifier.py` | Implementazione completa: modello, training, valutazione, predizione |
| `main_classifier.py` | CLI entrypoint per training da riga di comando |
| `notebooks/classifier.ipynb` | Notebook interattivo con pipeline completa |
| `config/config.yaml` | Configurazione (addestrata per v2.0) |

### Output

| File | Descrizione |
|------|-----------|
| `outputs/cnn_classifier.pt` | Modello addestrato (checkpoint) |
| `notebooks/outputs/confusion_matrix.png` | Matrice di confusione |
| `notebooks/outputs/roc_curve.png` | Curva ROC |
| `notebooks/outputs/` | Altri output del notebook |

### Come eseguire

**Da notebook (consigliato):**
```bash
jupyter notebook notebooks/classifier.ipynb
```

**Da CLI:**
```bash
python3 main_classifier.py --type1 data/processed/Waveforms_type1_extended_128samples.txt \
                           --type2 data/processed/Waveforms_type2_extended_128samples.txt \
                           --epochs 50 --lr 1e-3 --device auto
```

### Dati

| Percorso | Descrizione |
|----------|-------------|
| `data/processed/Waveforms_type1_extended_128samples.txt` | Training type1 (~7.2MB, ~100k waveforms) |
| `data/processed/Waveforms_type2_extended_128samples.txt` | Training type2 (~5.3MB, ~75k waveforms) |
| `data/measured/VLPs-2019_low-pass.dat` | Dati reali da classificare (~115MB, ~150k waveforms) |

---

## v1.0 — Autoencoder 1D (deprecato)

**Data:** Giugno 2025

### Cosa faceva

Addestrava due autoencoder 1D separati (AE1 su type1, AE2 su type2) e usava l'errore di ricostruzione come score di discriminazione su un file di waveform reali.

### File chiave (v1.0)

| File | Descrizione |
|------|-----------|
| `src/models.py` | `Conv1dAutoencoder` |
| `src/train.py` | Training loop autoencoder |
| `src/discriminator.py` | Scoring su file misurato |
| `notebooks/strainmeter_workflow.ipynb` | Notebook workflow v1.0 |
| `main_train.py` | CLI entrypoint v1.0 |

### Perché è stato sostituito

L'approccio autoencoder si basa su un'ipotesi implicita: le waveforms di un certo tipo saranno ricostruite meglio dall'autoencoder addestrato su quello stesso tipo. Questo è un approccio **non supervisionato** che non sfrutta le etichette di classe durante il training.

Il classificatore supervisionato v2.0:
- Usa direttamente le etichette (0=type1, 1=type2)
- Apprende discriminazioni più complesse e mirate
- Fornisce metriche standard (accuracy, AUC, ROC)
- È più interpretabile e facile da valutare

### Output v1.0 (conservati)

| Percorso | Descrizione |
|----------|-------------|
| `outputs/ae1_type1/` | Checkpoint AE1 |
| `outputs/ae2_type2/` | Checkpoint AE2 |
| `outputs/discriminator/` | Timeline di discriminazione |

---

## Note

- I file legacy di v1.0 sono ancora presenti nel repository per riferimento storico
- I file `src/autoencoder.py`, `src/data_processor.py`, `src/waveform_discriminator.py` sono alias di compatibilità e non devono essere usati per nuovo lavoro
- Per nuovi sviluppi, usare il codice di v2.0: `src/classifier.py`, `main_classifier.py`, `notebooks/classifier.ipynb`
