#!/bin/bash

# --- Configuración de Archivo Única ---
# Define la RUTA COMPLETA del archivo de entrada, incluyendo la extensión,
# dentro del directorio 'in/'. 
# Ejemplo: "in/among_us.png" o "in/mi_foto.jpg"
INPUT_FILE_FULL_PATH="in/among_us.png" # << CAMBIA ESTO >>

# --- Detección Automática de Rutas ---
# 1. Verificar si el archivo existe.
if [ ! -f "$INPUT_FILE_FULL_PATH" ]; then
    echo "❌ Error: El archivo de entrada '$INPUT_FILE_FULL_PATH' no existe."
    exit 1
fi

# 2. Extraer el nombre base (sin extensión ni 'in/').
FILENAME=$(basename "$INPUT_FILE_FULL_PATH")
INPUT_NAME="${FILENAME%.*}" 
OUTPUT_DIR="poster_${INPUT_NAME}_A2"

# 3. Detección automática del dispositivo SAM (asume 'cuda' si está disponible).
if command -v nvidia-smi &> /dev/null
then
    SAM_DEVICE="cuda"
else
    SAM_DEVICE="cpu"
fi
echo "🧠 Dispositivo de procesamiento SAM detectado: $SAM_DEVICE"


# --- Configuración para Poster A2 Landscape (300 DPI) ---
# A2 Landscape (420x594mm) a 300 DPI = 7016 x 4961 px.
MAX_WIDTH_A2=7016 # El lado más largo
PYTHON_SCRIPT="app.py" # Renombrado de app_gpu_poster.py a app.py


# --- Ejecución del Comando Principal ---
echo "⚙️ Iniciando proceso para ${INPUT_NAME} con configuración A2 Poster..."
python "$PYTHON_SCRIPT" \
    --input "$INPUT_FILE_FULL_PATH" \
    --out "$OUTPUT_DIR" \
    --sam-checkpoint sam_vit_b_01ec64.pth \
    --sam-model "vit_b" \
    --max-width $MAX_WIDTH_A2 \
    \
    # Configuración de Auto-K (Aumento de exigencia)
    --auto-k --k-min 12 --k-max 24 \
    --target-ssim 0.965 \
    --max-small-ratio 0.01 \
    \
    # Configuración de Bordes y Áreas (Escalado A2)
    --line-thickness 2 \
    --edge-deltaE 3.5 \
    --slic-n 8000 \
    --slic-compact 8.0 \
    --smooth-open 0 \
    --smooth-close 1 \
    --min-region-area 100 \
    --numbers-min-area 50 \
    --font-size 20 \
    --close-gaps-radius 2 \
    --force-closed \
    \
    # Configuración de SAM (Usa el dispositivo detectado)
    --sam-device "$SAM_DEVICE" \
    --sam-pps 128 \
    --sam-min-area 1500 \
    --sam-iou 0.90 \
    --sam-stability 0.93 \
    \
    # Configuración de PDF (Fija el tamaño de página a A2 Landscape)
    --page-size A2 \
    --orientation landscape \
    --dpi 300

echo "✅ Proceso completado. Resultados en el directorio: ${OUTPUT_DIR}"