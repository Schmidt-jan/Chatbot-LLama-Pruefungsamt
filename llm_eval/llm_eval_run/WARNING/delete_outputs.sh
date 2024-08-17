#!/bin/bash

# Pfade zu den zu leerenden Verzeichnissen
plot_dir_base="/home/tpllmws23/Chatbot-LLama-Pruefungsamt/llm_eval/output/plots"
promptfoo_data_dir_base="/home/tpllmws23/Chatbot-LLama-Pruefungsamt/llm_eval/output/promptfoo_data"

# Funktion zum Leeren eines Verzeichnisses
empty_directory() {
    local dir_path=$1
    if [ -d "$dir_path" ]; then
        rm -rf "$dir_path"/*
        echo "Inhalt von '$dir_path' wurde gelöscht."
    else
        echo "Verzeichnis '$dir_path' existiert nicht."
    fi
}

# Verzeichnisse, deren Inhalte gelöscht werden sollen
empty_directory "$plot_dir_base"
empty_directory "$promptfoo_data_dir_base"

echo "Alle Inhalte der Plotverzeichnisse und des promptfoo_data-Ordners wurden gelöscht."
