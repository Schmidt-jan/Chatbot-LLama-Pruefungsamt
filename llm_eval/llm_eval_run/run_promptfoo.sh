#!/bin/bash

# Zielverzeichnis für die Ausgabe
output_dir="output/promptfoo_data/raw"

# Aktuellen Zeitstempel im Format 'yyyymmdd_HHMMSS' erstellen
timestamp=$(date +"%Y%m%d_%H%M%S")

# Dateiname mit Zeitstempel erstellen
output_filename="output_$timestamp.json"

source /home/tpllmws23/environments/rag/bin/activate

# Schritt 1: Verzeichnis wechseln (zum übergeordneten Verzeichnis)
cd ..

echo "Python Version:"
which python
python --version

# Schritt 2: promptfoo-Befehl ausführen und Ausgabe speichern im angegebenen Verzeichnis
promptfoo eval -j 1 --no-cache -o "$output_dir/$output_filename"

# Schritt 4: Warten bis die Ausgabedatei existiert
while [ ! -f "$output_dir/$output_filename" ]; do
    echo "Warte auf Ausgabedatei $output_filename..."
    sleep 1  # Eine Sekunde warten, bevor erneut überprüft wird
done

# Schritt 5: Wieder zurück in das llm_eval_run Verzeichnis wechseln
cd llm_eval_run

# Schritt 6: Wenn die Ausgabedatei existiert, das Python-Skript aufrufen
python3 output_parser.py $timestamp

# Schritt 7: Wenn die Ausgabedatei existiert, das Python-Skript aufrufen
python3 llm_eval_plotter.py $timestamp
