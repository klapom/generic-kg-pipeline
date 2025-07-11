#!/bin/bash

echo "📊 Überprüfung der extrahierten Daten"
echo "===================================="

OUTPUT_DIR="data/output"

if [ -d "$OUTPUT_DIR" ]; then
    echo "📁 Output-Verzeichnis existiert"
    
    # Anzahl der Dateien
    file_count=$(ls -1 "$OUTPUT_DIR" 2>/dev/null | wc -l)
    echo "📄 Dateien im Output: $file_count"
    
    # Liste der Dateien (ohne Inhalte)
    echo "📋 Vorhandene Dateien:"
    ls -la "$OUTPUT_DIR"
    
    # Größe der Dateien
    echo "📏 Dateigröße-Übersicht:"
    du -sh "$OUTPUT_DIR"/* 2>/dev/null || echo "Keine Dateien gefunden"
    
    # JSON-Struktur anzeigen (nur Schlüssel, nicht Werte)
    for file in "$OUTPUT_DIR"/*.json; do
        if [ -f "$file" ]; then
            echo "🔍 Struktur von $(basename "$file"):"
            jq -r 'paths(scalars) as $p | $p | join(".")' "$file" 2>/dev/null | head -20
            echo ""
        fi
    done
else
    echo "❌ Output-Verzeichnis existiert nicht"
fi