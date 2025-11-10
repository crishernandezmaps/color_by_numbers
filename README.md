# 🎨 Color-by-Numbers Generator (SAM + SLIC + ΔE2000)

Este es un proyecto personal desarrollado en Python para generar kits de "Pintar por Números" de alta calidad, utilizando técnicas avanzadas de Computer Vision. La clave es el balance entre los contornos estructurales proporcionados por el **Segment Anything Model (SAM) de Meta** y el detalle fino capturado por la segmentación **SLIC** y los umbrales de color **ΔE2000**.

## 🚀 Instalación y Configuración

Sigue estos pasos para configurar tu entorno y descargar los archivos necesarios.

### 1\. Entorno Virtual y Dependencias

Es altamente recomendable usar un entorno virtual (`venv`):

```bash
# Crear y activar el entorno
python3 -m venv venv
source venv/bin/activate

# Instalar las dependencias (requiere PyTorch, scikit-image, OpenCV, PIL, ReportLab)
pip install -r requirements.txt
```

> **Nota:** El archivo `requirements.txt` debe contener todas las bibliotecas utilizadas en el script (`numpy`, `pillow`, `opencv-python`, `scikit-image`, `reportlab`, `torch`, `segment-anything`, etc.).

### 2\. Descarga del Modelo SAM (Checkpoint)

Debes descargar el *checkpoint* del modelo SAM (`vit_b`). Este archivo es grande (aproximadamente 375 MB) y **está excluido del repositorio** por medio del archivo `.gitignore`.

Descárgalo y colócalo en el directorio raíz de tu proyecto:

```bash
# Ejemplo de descarga (usando wget o cURL, si está disponible)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# O usa el script 'pth_down.sh' si lo tienes configurado.
```

## ⚙️ Uso del Script

El script principal es `new.py`. Coloca la imagen de entrada (ej. `frida02.png`) en la carpeta **`in/`**.

El formato de uso es:

```bash
python new.py --input "in/NOMBRE.png" --out NOMBRE_SALIDA [OPCIONES...]
```

### Ejemplo de Comando Optimizado (Retrato Frida Kahlo)

Este comando es el resultado de varias iteraciones, logrando un balance óptimo entre detalle y contornos limpios.

```bash
python new.py \
    --input "in/frida02.png" \
    --out frida02 \
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
    --orientation portrait
```

-----

## 🔬 Explicación de Parámetros Clave

Los parámetros determinan la calidad y la complejidad del kit.

### Paleta de Colores (K-Means)

| Parámetro | Propósito | Valor Optimizado (Frida) |
| :--- | :--- | :--- |
| **`--auto-k`** | Activa la búsqueda automática del número óptimo de colores (K). | `True` |
| **`--k-min`/`--k-max`** | Rango de colores a buscar. | `12` / `24` |
| **`--target-ssim`** | Métrica de calidad: qué tan fiel debe ser la imagen final a la original. Un valor más alto fuerza más colores. | `0.965` |

### Segmentación de Regiones (SLIC y Morfología)

| Parámetro | Propósito | Valor Optimizado (Frida) |
| :--- | :--- | :--- |
| **`--slic-n`** | Número de *superpíxeles* SLIC. Más alto = más regiones iniciales para delinear el detalle fino. | `4000` |
| **`--slic-compact`** | Peso que se le da a la cercanía espacial vs. el color. Menor valor (ej. `8.0`) respeta más las diferencias de color. | `8.0` |
| **`--min-region-area`** | Área mínima (en píxeles) para que una región sea considerada válida y no fusionada. | `30` |
| **`--max-small-ratio`** | Máximo porcentaje de la imagen permitido en regiones demasiado pequeñas para ser numeradas. | `0.01` |

### Bordes y Contornos

| Parámetro | Propósito | Valor Optimizado (Frida) |
| :--- | :--- | :--- |
| **`--edge-deltaE`** | **Umbral de Borde.** Mide la diferencia de color (Delta E 2000) entre dos regiones adyacentes. Un valor más bajo genera más líneas. | `3.5` |
| **`--force-closed`** | **Estructural.** Asegura que todos los contornos estén cerrados, previniendo "fugas" de color. | `True` |

### Segment Anything Model (SAM)

| Parámetro | Propósito | Valor Optimizado (Frida) |
| :--- | :--- | :--- |
| **`--sam-device`** | Dispositivo a usar para Torch (CPU, cuda, mps). Se recomienda `cpu` para modelos pequeños si no hay GPU. | `cpu` |
| **`--sam-min-area`** | Área mínima (en píxeles) para que SAM genere una máscara (elimina ruido). | `400` |
| **`--sam-pps`** | Puntos por lado. Cuantos más puntos, más oportunidades tiene SAM de detectar un objeto. | `128` |