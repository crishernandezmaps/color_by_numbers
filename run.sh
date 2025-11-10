#!/bin/bash

# --- Configuración de Archivo ---
# Define el nombre base del archivo, sin la extensión ni el directorio 'in/'.
# Ejemplo: si el archivo de entrada es 'in/frida02.png', usa 'frida02'.
INPUT_NAME="among_us"
EXT="png"

# Construye las rutas completas
INPUT_FILE="in/${INPUT_NAME}.${EXT}"
OUTPUT_DIR="${INPUT_NAME}" # Esto se mapea directamente al argumento --out

# Verifica si el archivo de entrada existe
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: El archivo de entrada '$INPUT_FILE' no existe."
    exit 1
fi

# --- Ejecución del Comando Principal ---
echo "⚙️ Iniciando proceso para ${INPUT_NAME}..."
python app.py \
    --input "$INPUT_FILE" \
    --out "$OUTPUT_DIR" \
    --sam-checkpoint sam_vit_b_01ec64.pth \
    --sam-model "vit_b" \
    --max-width 4961 \
    --auto-k --k-min 12 --k-max 24 \
    --target-ssim 0.965 \
    --max-small-ratio 0.01 \
    --line-thickness 1 \
    --edge-deltaE 3.5 \
    --slic-n 4000 \
    --slic-compact 8.0 \
    --smooth-open 0 \
    --smooth-close 1 \
    --min-region-area 30 \
    --numbers-min-area 20 \
    --font-size 14 \
    --sam-device cpu \
    --sam-pps 128 \
    --sam-min-area 400 \
    --sam-iou 0.90 \
    --sam-stability 0.93 \
    --force-closed \
    --close-gaps-radius 1 \
    --orientation landscape

echo "✅ Proceso completado. Resultados en el directorio: ${OUTPUT_DIR}"