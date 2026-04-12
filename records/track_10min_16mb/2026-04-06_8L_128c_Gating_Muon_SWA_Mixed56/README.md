# Parameter Golf Final Submission: Legend 128C

Diese Repository enthält die finale Einreichung für die Parameter Golf Challenge. Das Modell wurde gezielt für den Fineweb-10B Datensatz unter Einhaltung der offiziellen 10-Minuten-Trainingszeit und des 16MB Speicherlimits optimiert.

## Technische Spezifikationen

- **Architektur**: 8-Layer Transformer mit 640 Hidden Dimension (10 Attention Heads).
- **Specialist Core**: 128-Cluster Specialist Gating (Legend 128C) für hochpräzise Token-Vorhersage.
- **Optimierung**: Muon Matrix Optimizer (LR 0.02) für Gewichtsmatrizen + Adam Optimizer für Skalare und Embeddings.
- **Lernrate**: Warmup (100 Iterationen) und Warmdown (3500 Iterationen) bis auf Null.
- **Quantisierung**: Mixed-bit Quantisierung (uint5/uint6/int8) zur Einhaltung des 16MB Limits bei maximaler Ergebnisqualität.

## Ergebnisse (Verifiziert auf 8x H100 Cluster)

Die Ergebnisse basieren auf dem finalen Trainingslauf (siehe `training_log.txt`):

- **Training BPB (SWA)**: **1.2251** (bei Step 9200)
- **Quantisierte BPB (Roundtrip)**: **1.2331**
- **Trainingszeit**: ~616 Sekunden (Training + Validierung + Quantisierung).
- **Dateigröße**: **13.00 MB** (Komprimiertes Modell + Code).

## Inhalt der Einreichung

1. **`train_gpt.py`**: Das vollständige Trainings- und Quantisierungs-Skript.
2. **`final_model.int8.ptz`**: Das finale Modell-Artefakt (zlib-komprimiert).
3. **`training_log.txt`**: Detailliertes Log des offiziellen H100 Cluster-Runs.

## Anleitung zur Reproduktion

1. Sicherstellen, dass die `DATA_PATH` und `TOKENIZER_PATH` Umgebungsvariablen korrekt gesetzt sind.
2. Das Training auf einem 8x H100 System starten:
   ```bash
   torchrun --nproc_per_node=8 train_gpt.py
   ```
3. Das Skript führt nach dem Training automatisch die SWA-Gewichtung, Mixed-bit Quantisierung und eine finale Validierung durch.

