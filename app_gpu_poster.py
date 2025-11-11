#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Color-by-Numbers + SAM (Segment Anything) + SLIC + ΔE2000 edges
- SAM: macro-bordes coherentes.
- K-Means: paleta.
- SLIC + ΔE2000: microdetalles controlados por umbral.
- Números: stroke + tamaño fijo, con lógica de reintento para asegurar cobertura.
- Contornos: Fallback estructural para garantizar el cierre de zonas (--force-closed).
"""

import argparse, io, csv
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

from skimage import measure, morphology, color as skcolor, segmentation
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import distance_transform_edt as dist
from skimage.segmentation import find_boundaries # <-- Nueva

# PDF
from reportlab.lib.pagesizes import A0, A1, A2, A3, A4, landscape, portrait
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import cm, mm

# Torch / SAM
import torch
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────
def _pick_device(s: str) -> torch.device:
    s = s.lower()
    if s == "auto":
        if torch.backends.mps.is_available(): return torch.device("mps")
        if torch.cuda.is_available(): return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(s)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def imread_any(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))

def save_png(arr: np.ndarray, path: Path):
    Image.fromarray(arr).save(path, "PNG")

def to_hex(rgb_tuple):
    return "#{:02X}{:02X}{:02X}".format(*[int(x) for x in rgb_tuple])

def resize_max_width(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w: return img
    scale = max_w / float(w)
    return cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)

def sort_palette_by_lightness(centroids_rgb: np.ndarray) -> np.ndarray:
    lab = skcolor.rgb2lab(centroids_rgb.reshape(1, -1, 3)).reshape(-1, 3)
    return np.argsort(lab[:, 0])

def kmeans_quantize(img_rgb: np.ndarray, k: int, attempts=3, sample=1.0, seed=42):
    H, W = img_rgb.shape[:2]
    flat = img_rgb.reshape(-1, 3).astype(np.float32)
    if sample < 1.0:
        n = max(1000, int(len(flat) * sample))
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(flat), size=n, replace=False)
        data_for_fit = flat[idx]
    else:
        data_for_fit = flat
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _compact, _labels, centers = cv2.kmeans(
        data_for_fit, K=k, bestLabels=None, criteria=criteria,
        attempts=attempts, flags=cv2.KMEANS_PP_CENTERS
    )
    dists = np.sum((flat[:, None, :] - centers[None, :, :])**2, axis=2)
    labels_full = np.argmin(dists, axis=1).astype(np.int32)
    labels_img = labels_full.reshape(H, W)
    centroids_rgb = np.clip(centers, 0, 255).astype(np.uint8)
    return labels_img, centroids_rgb

def smooth_labels(labels: np.ndarray, radius_open=1, radius_close=1) -> np.ndarray:
    classes = np.unique(labels)
    H, W = labels.shape
    accum = np.zeros((H, W, len(classes)), dtype=np.uint8)
    for i, c in enumerate(classes):
        mask = (labels == c).astype(np.uint8)
        if radius_open > 0:
            se = morphology.disk(radius_open)
            mask = morphology.opening(mask, se)
        if radius_close > 0:
            se = morphology.disk(radius_close)
            mask = morphology.closing(mask, se)
        accum[:, :, i] = mask
    out = np.argmax(accum, axis=2)
    return classes[out]

def _load_font(font_path: Optional[str], font_size: int):
    # robust on macOS/Linux
    candidates = []
    if font_path: candidates.append(font_path)
    candidates += [
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Helvetica.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            if Path(p).exists():
                return ImageFont.truetype(p, font_size)
        except Exception:
            pass
    return ImageFont.load_default()

def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        l,t,r,b = draw.textbbox((0,0), text, font=font); return r-l, b-t
    if hasattr(draw, "textlength"):
        w = int(draw.textlength(text, font=font))
        try: h = int(font.getbbox(text)[3] - font.getbbox(text)[1])
        except Exception: h = int(getattr(font, "size", 12) * 1.2)
        return w, h
    if hasattr(draw, "textsize"):
        return draw.textsize(text, font=font)
    return (len(text) * int(getattr(font, "size", 12) * 0.6), int(getattr(font, "size", 12) * 1.2))


# ──────────────────────────────────────────────────────────────────────────────
# Numeración (con reintentos y auditoría)
# ──────────────────────────────────────────────────────────────────────────────
def place_numbers_on_regions(
    base_outline: Image.Image,
    labels: np.ndarray,
    order_map: Dict[int,int],
    min_area=80,
    font_size=14,
    safety_margin=2,
    font_path=None,
    stroke_width=1,
    stroke_fill=(255,255,255),
) -> Image.Image:
    """
    Dibuja solo el número (sin círculo detrás), con lógica de reintento para asegurar cobertura
    en regiones estrechas y una auditoría final para numerar los que falten.
    """
    draw = ImageDraw.Draw(base_outline)
    base_font = _load_font(font_path, font_size)
    font = base_font # Se usa tamaño fijo
    
    numbered_cc = set()

    for c in np.unique(labels):
        lab = measure.label((labels == c).astype(np.uint8), connectivity=1)
        for r in measure.regionprops(lab):
            if r.area < min_area:
                continue

            mask = (lab == r.label)

            # 1) Punto de máxima distancia (más gordo)
            d0 = dist(mask)
            cy, cx = np.unravel_index(np.argmax(d0), d0.shape)
            best_pt = (int(cx), int(cy))
            best_d  = float(d0[cy, cx])
            
            # 2) Si es muy angosto (<1.5), intenta centroide
            if best_d < 1.5:
                cy2, cx2 = int(r.centroid[0]), int(r.centroid[1])
                # comprueba que el centroide caiga dentro de la máscara (para evitar bordes)
                if mask[min(max(cy2,0),mask.shape[0]-1), min(max(cx2,0),mask.shape[1]-1)]:
                    best_pt = (cx2, cy2)
                    best_d  = float(d0[cy2, cx2])

            # 3) Si sigue angosto, erode y reintenta
            if best_d < 1.5:
                se = morphology.disk(1)
                mask_er = morphology.erosion(mask, se)
                if mask_er.any():
                    d1 = dist(mask_er)
                    cy3, cx3 = np.unravel_index(np.argmax(d1), d1.shape)
                    # si la nueva distancia es mejor
                    if d1[cy3, cx3] > best_d:
                        best_pt = (int(cx3), int(cy3))
                        best_d  = float(d1[cy3, cx3])

            # 4) Si todavía es muy fino, cae al centroide “sí o sí”
            if best_d < 1.0:
                cy4, cx4 = int(r.centroid[0]), int(r.centroid[1])
                best_pt = (cx4, cy4)
            
            x, y = best_pt

            # Evita márgenes ultra estrechos (aunque con el reintento ya es más robusto)
            if (r.bbox[3]-r.bbox[1] <= safety_margin) or (r.bbox[2]-r.bbox[0] <= safety_margin):
                # Aún así, usamos el centroide para el fallback
                y, x = int(r.centroid[0]), int(r.centroid[1])

            text = str(order_map.get(int(c), int(c)) + 1)
            w, h = _text_size(draw, text, font)
            
            draw.text(
                (x - w/2, y - h/2),
                text,
                fill=(0,0,0),
                font=font,
                stroke_width=int(stroke_width),
                stroke_fill=stroke_fill,
            )
            numbered_cc.add((int(c), int(r.label)))

    # Pass de auditoría: coloca número en centroides de los no-numerados
    for c in np.unique(labels):
        lab = measure.label((labels == c).astype(np.uint8), connectivity=1)
        for r in measure.regionprops(lab):
            if r.area < min_area:
                continue
            key = (int(c), int(r.label))
            if key in numbered_cc:
                continue
            
            # Coloca en el centroide como fallback final
            cy, cx = int(r.centroid[0]), int(r.centroid[1])
            text = str(order_map.get(int(c), int(c)) + 1)
            w, h = _text_size(draw, text, font)
            
            draw.text(
                (cx - w/2, cy - h/2),
                text,
                fill=(0,0,0),
                font=font,
                stroke_width=int(stroke_width),
                stroke_fill=stroke_fill
            )

    return base_outline


# ──────────────────────────────────────────────────────────────────────────────
# SAM helpers
# ──────────────────────────────────────────────────────────────────────────────
def run_sam_masks(img_rgb: np.ndarray, checkpoint: Path, model_name: str, device: torch.device,
                  points_per_side=64, pred_iou_thresh=0.88, stability_score_thresh=0.92,
                  min_mask_region_area=5000) -> List[np.ndarray]:
    sam = sam_model_registry[model_name](checkpoint=str(checkpoint))
    sam.to(device=device)
    gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area
    )
    masks = gen.generate(img_rgb)
    masks = sorted(masks, key=lambda m: m['area'], reverse=True)
    return [m["segmentation"] for m in masks]

def labels_modal_within_masks(labels: np.ndarray, masks: List[np.ndarray], modal_frac: float = 0.7) -> np.ndarray:
    out = labels.copy()
    for seg in masks:
        region = labels[seg]
        vals, counts = np.unique(region, return_counts=True)
        j = int(np.argmax(counts))
        if counts[j] / float(region.size) >= modal_frac:
            out[seg] = vals[j]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Bordes ΔE + SAM con Fallback Estructural (find_boundaries)
# ──────────────────────────────────────────────────────────────────────────────
def boundary_map_deltaE(labels: np.ndarray, centroids_rgb: np.ndarray, deltaE_thresh: float) -> np.ndarray:
    H, W = labels.shape
    lab_palette = rgb2lab((centroids_rgb.astype(np.float32)/255.0)[None, :, :]).reshape(-1, 3)
    b = np.zeros((H, W), dtype=bool)
    diff = labels[1:, :] != labels[:-1, :]
    if diff.any():
        a = labels[1:, :][diff]; c = labels[:-1, :][diff]
        de = deltaE_ciede2000(lab_palette[a], lab_palette[c])
        mask = np.zeros_like(diff, dtype=bool); mask[diff] = (de >= deltaE_thresh)
        b[1:, :] |= mask; b[:-1, :] |= mask
    diff = labels[:, 1:] != labels[:, :-1]
    if diff.any():
        a = labels[:, 1:][diff]; c = labels[:, :-1][diff]
        de = deltaE_ciede2000(lab_palette[a], lab_palette[c])
        mask = np.zeros_like(diff, dtype=bool); mask[diff] = (de >= deltaE_thresh)
        b[:, 1:] |= mask; b[:, :-1] |= mask
    return b

def outline_from_labels(labels: np.ndarray, centroids_rgb: np.ndarray, deltaE_thresh: float,
                        sam_masks: Optional[List[np.ndarray]], line_thickness: int,
                        close_gaps_radius: int = 2, force_closed: bool = False) -> np.ndarray:
    """
    Genera el contorno, combinando ΔE, SAM y un fallback estructural para garantizar el cierre.
    """
    H, W = labels.shape
    canvas_img = np.full((H, W, 3), 255, dtype=np.uint8)

    # 1) Bordes por ΔE entre clases adyacentes
    e_delta = boundary_map_deltaE(labels, centroids_rgb, deltaE_thresh)

    # 2) Bordes de SAM
    e_sam = np.zeros((H, W), dtype=bool)
    if sam_masks:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        for seg in sam_masks:
            edge = cv2.morphologyEx(seg.astype(np.uint8)*255, cv2.MORPH_GRADIENT, k) > 0
            e_sam |= edge

    # 3) Fallback estructural: bordes cerrados del mapa de etiquetas
    e_struct = find_boundaries(labels, mode="outer")  # cerrado por definición

    # 4) Combinación:
    e_raw = (e_delta | e_sam)
    if force_closed:
        e_raw |= e_struct # Si se fuerza, añadimos la estructura base

    # 5) Cerrar micro-grietas y depurar
    kclose = morphology.disk(max(1, int(close_gaps_radius)))
    eb = morphology.binary_closing(e_raw, kclose)
    eb = morphology.remove_small_holes(eb, area_threshold=64)

    # 6) Afinar y grosor final
    eb = morphology.thin(eb)  # esqueleto 1px, garantizando continuidad
    edges = (eb.astype(np.uint8) * 255)

    if line_thickness > 1:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_thickness, line_thickness))
        edges = cv2.dilate(edges, k2)

    canvas_img[edges > 0] = (0, 0, 0)
    return canvas_img


# ──────────────────────────────────────────────────────────────────────────────
# Paleta & PDF (Métricas de Auto-K incluidas aquí para su correcta definición)
# ──────────────────────────────────────────────────────────────────────────────
def make_palette_image(centroids_rgb: np.ndarray, order: np.ndarray, swatch_w=260, swatch_h=60, pad=12):
    K = len(centroids_rgb)
    W = swatch_w
    H = K * (swatch_h + pad) + pad
    img = Image.new("RGB", (W, H), (255, 255, 255))
    drw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("DejaVuSans.ttf", 18)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font_title = ImageFont.load_default(); font_small = ImageFont.load_default()
    y = pad
    drw.text((pad, y), "Paleta (1 = claro → n = oscuro)", fill=(0,0,0), font=font_title)
    y += 28
    for i, idx in enumerate(order):
        rgb = tuple(int(x) for x in centroids_rgb[idx])
        drw.rectangle([pad, y, pad + swatch_h, y + swatch_h], fill=rgb, outline=(0,0,0))
        drw.text((pad + swatch_h + 10, y + 10), f"{i+1:>2}  RGB{rgb}  {to_hex(rgb)}", fill=(0,0,0), font=font_small)
        y += swatch_h + pad
    return np.array(img)

def _pagesize_from_args(size_name: str, w_mm: float, h_mm: float, orientation: str) -> Optional[Tuple[float, float]]:
    preset = {"A0": A0, "A1": A1, "A2": A2, "A3": A3, "A4": A4}.get(size_name.upper())
    if size_name == "custom" and w_mm > 0 and h_mm > 0:
        preset = (w_mm*mm, h_mm*mm)
    if size_name == "auto" or preset is None:
        return None # 'auto': usa el tamaño del primer PNG
    return landscape(preset) if orientation=="landscape" else portrait(preset)

def export_pdf(a4_png_paths: List[Path], out_pdf: Path, dpi: int = 300, page_size_name: str = "A2",
               orientation: str = "portrait", page_wmm: float = 0.0, page_hmm: float = 0.0):
    
    ps = _pagesize_from_args(page_size_name, page_wmm, page_hmm, orientation)
    
    if ps is None:
        # tamaño = del primer PNG (no escalar, usar DPI)
        pil0 = Image.open(a4_png_paths[0]).convert("RGB")
        # define la página en puntos (1 punto = 1/72 de pulgada) a razón de DPI
        page_w = pil0.width * 72.0/ dpi
        page_h = pil0.height * 72.0/ dpi
        page_size = (page_w, page_h)
    else:
        page_size = ps

    c = canvas.Canvas(str(out_pdf), pagesize=page_size)
    page_w, page_h = page_size

    # margen pequeño o cero para poster (usaremos 0.5 cm como mínimo)
    margin = 0.5 * cm
    usable_w = page_w - 2*margin
    usable_h = page_h - 2*margin
    
    # Si la página es 'auto', el margen es 0
    if ps is None: margin = 0
    
    for p in a4_png_paths:
        pil = Image.open(p).convert("RGB")
        w, h = pil.size

        # Si 'auto', no reescales; si preset, ajusta a página manteniendo proporción:
        if ps is None:
            new_w = page_w
            new_h = page_h
        else:
            scale = min(usable_w / w, usable_h / h)
            new_w, new_h = int(w*scale), int(h*scale)

        try:
            resample = Image.Resampling.LANCZOS
        except Exception:
            resample = Image.LANCZOS
        pil = pil.resize((new_w, new_h), resample)

        bio = io.BytesIO(); pil.save(bio, format="PNG"); bio.seek(0)
        
        # Centrado en la página:
        x = margin + (usable_w - new_w)/2
        y = margin + (usable_h - new_h)/2
        
        # Si ps es None (tamaño auto), no hay usable_w, solo page_w
        if ps is None:
            x = (page_w - new_w)/2
            y = (page_h - new_h)/2

        c.drawImage(ImageReader(bio), x, y, width=new_w, height=new_h)
        c.showPage()
    c.save()


def _region_metrics(labels: np.ndarray, min_region_area: int) -> Tuple[int, float]:
    H, W = labels.shape; total = H*W
    small_pixels = 0; count_regions = 0
    for c in np.unique(labels):
        lab = measure.label((labels == c).astype(np.uint8), connectivity=1)
        for r in measure.regionprops(lab):
            if r.area < min_region_area: small_pixels += r.area
            else: count_regions += 1
    return count_regions, small_pixels / float(total)

def _ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean([ssim(a[..., i], b[..., i], data_range=255) for i in range(3)]))

def choose_optimal_k(img_rgb: np.ndarray, k_min: int, k_max: int, smooth_open: int, smooth_close: int,
                     min_region_area: int, target_ssim: float, max_small_ratio: float,
                     max_nregions: Optional[int], sample: float) -> int:
    best_k = k_max
    base = cv2.bilateralFilter(img_rgb, d=7, sigmaColor=40, sigmaSpace=40)
    for k in range(k_min, k_max + 1):
        labels, cents = kmeans_quantize(base, k=k, sample=sample)
        order = sort_palette_by_lightness((cents.astype(np.float32)/255.).reshape(-1,1,3).squeeze(1))
        inv = np.empty_like(order); inv[order] = np.arange(len(order))
        labels_num = inv[labels]
        labels_smooth = smooth_labels(labels_num, radius_open=smooth_open, radius_close=smooth_close)
        ref = cents[order][labels_smooth]
        q_ssim = _ssim_rgb(base, ref)
        n_regions, small_ratio = _region_metrics(labels_smooth, min_region_area)
        if (q_ssim >= target_ssim) and (small_ratio <= max_small_ratio) and (max_nregions is None or n_regions <= max_nregions):
            best_k = k; break
    return int(best_k)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Color-by-Numbers con SAM + SLIC + ΔE edges")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="outkit")
    ap.add_argument("--max-width", type=int, default=2200)
    ap.add_argument("--colors", type=int, default=12)
    ap.add_argument("--auto-k", action="store_true")
    ap.add_argument("--k-min", type=int, default=10)
    ap.add_argument("--k-max", type=int, default=24)
    ap.add_argument("--target-ssim", type=float, default=0.92)
    ap.add_argument("--max-small-ratio", type=float, default=0.06)
    ap.add_argument("--max-nregions", type=int, default=0)
    ap.add_argument("--smooth-open", type=int, default=1)
    ap.add_argument("--smooth-close", type=int, default=2)
    ap.add_argument("--min-region-area", type=int, default=220)
    ap.add_argument("--numbers-min-area", type=int, default=60)
    ap.add_argument("--numbers-scale", type=float, default=1.0) # Se ignora, pero se mantiene por compatibilidad
    ap.add_argument("--line-thickness", type=int, default=1)
    ap.add_argument("--font-size", type=int, default=12)
    ap.add_argument("--font-path", type=str, default="")
    ap.add_argument("--sample", type=float, default=0.9)
    ap.add_argument("--orientation", type=str, default="portrait", choices=["portrait","landscape"], help="Orientación de la página en el PDF final.")
    
    # PARÁMETROS para el tamaño de página (impresión)
    ap.add_argument("--page-size", type=str, default="A4",
                    choices=["A0","A1","A2","A3","A4","auto","custom"],
                    help="Tamaño de página del PDF final (usar 'auto' si no se quiere reescalar).")
    ap.add_argument("--page-width-mm", type=float, default=0.0, help="Ancho de página en mm para custom.")
    ap.add_argument("--page-height-mm", type=float, default=0.0, help="Alto de página en mm para custom.")
    
    # NUEVOS PARÁMETROS para la continuidad de los bordes
    ap.add_argument("--close-gaps-radius", type=int, default=2, help="Radio de cierre morfológico aplicado a los bordes antes de afinar.")
    ap.add_argument("--force-closed", action="store_true", help="Si se activa, se usa find_boundaries para garantizar que todos los contornos estén cerrados.")

    # SLIC
    ap.add_argument("--slic-n", type=int, default=1800)
    ap.add_argument("--slic-compact", type=float, default=12.0)
    # ΔE
    ap.add_argument("--edge-deltaE", type=float, default=7.0)
    # SAM
    ap.add_argument("--sam-checkpoint", type=str, required=True)
    ap.add_argument("--sam-model", type=str, default="vit_b", choices=["vit_b","vit_l","vit_h"])
    ap.add_argument("--sam-min-area", type=int, default=5000)
    ap.add_argument("--sam-pps", type=int, default=64)
    ap.add_argument("--sam-iou", type=float, default=0.88)
    ap.add_argument("--sam-stability", type=float, default=0.92)
    ap.add_argument("--sam-device", type=str, default="auto", choices=["auto","cpu","mps","cuda"])
    ap.add_argument("--dpi", type=int, default=300, help="DPI usado para el cálculo del tamaño de página si --page-size es 'auto'.")


    args = ap.parse_args()

    in_path = Path(args.input); out_dir = Path(args.out)
    if not in_path.exists(): raise FileNotFoundError(in_path)
    ensure_dir(out_dir)

    device = _pick_device(args.sam_device)
    print(f"🧠 Torch device: {device}")

    rgb = resize_max_width(imread_any(in_path), args.max_width)

    # Auto-K
    max_nregions = None if args.max_nregions <= 0 else args.max_nregions
    if args.auto_k:
        chosen_k = choose_optimal_k(
            img_rgb=rgb, k_min=args.k_min, k_max=args.k_max,
            smooth_open=args.smooth_open, smooth_close=args.smooth_close,
            min_region_area=args.min_region_area, target_ssim=args.target_ssim,
            max_small_ratio=args.max_small_ratio, max_nregions=max_nregions,
            sample=args.sample
        )
        print(f"🎯 Auto-K → {chosen_k}")
        args.colors = chosen_k

    # Pre-suavizado y K-Means
    rgb_b = cv2.bilateralFilter(rgb, d=7, sigmaColor=40, sigmaSpace=40)
    labels, centroids = kmeans_quantize(rgb_b, k=args.colors, sample=args.sample)

    # Reordenado por L* y remap a [0..K-1]
    order = sort_palette_by_lightness((centroids.astype(np.float32)/255.).reshape(-1,1,3).squeeze(1))
    inv_order = np.empty_like(order); inv_order[order] = np.arange(len(order))
    labels_num = inv_order[labels]

    # Suavizado morfológico + SLIC
    labels_smooth = smooth_labels(labels_num, radius_open=args.smooth_open, radius_close=args.smooth_close)
    if args.slic_n > 0:
        seg = segmentation.slic(
            rgb_b, n_segments=args.slic_n, compactness=args.slic_compact,
            start_label=0, channel_axis=-1
        )
        tmp = labels_smooth.copy()
        for sp in range(seg.max()+1):
            m = seg == sp
            if m.any(): # Asegura que el segmento no esté vacío
                vals, cnt = np.unique(tmp[m], return_counts=True)
                labels_smooth[m] = vals[np.argmax(cnt)]

    # SAM + modal label con umbral
    sam_masks = run_sam_masks(
        img_rgb=rgb_b,
        checkpoint=Path(args.sam_checkpoint),
        model_name=args.sam_model,
        device=device,
        points_per_side=args.sam_pps,
        pred_iou_thresh=args.sam_iou,
        stability_score_thresh=args.sam_stability,
        min_mask_region_area=args.sam_min_area
    )
    labels_sam_smooth = labels_modal_within_masks(labels_smooth, sam_masks, modal_frac=0.7)

    # Imagen referencia y contornos
    centroids_ordered = centroids[order]
    ref = centroids_ordered[labels_sam_smooth]

    outline = outline_from_labels(
        labels=labels_sam_smooth,
        centroids_rgb=centroids_ordered,
        deltaE_thresh=args.edge_deltaE,
        sam_masks=sam_masks,
        line_thickness=args.line_thickness,
        close_gaps_radius=args.close_gaps_radius,
        force_closed=args.force_closed
    )

    # Numeración (con reintentos y auditoría)
    outline_pil = Image.fromarray(outline.copy())
    outline_numbered = np.array(
        place_numbers_on_regions(
            base_outline=outline_pil,
            labels=labels_sam_smooth,
            order_map={int(i): int(i) for i in range(len(centroids_ordered))},
            min_area=args.numbers_min_area,
            font_size=args.font_size,
            font_path=args.font_path if args.font_path else None,
            stroke_width=1,
            stroke_fill=(255, 255, 255),
        )
    )

    # Paleta
    palette_img = make_palette_image(centroids_ordered, order=np.arange(len(centroids_ordered)))

    # Guardados y PDF
    outline_path = out_dir / "01_outline_numbered.png"
    ref_path     = out_dir / "02_colored_reference.png"
    palette_path = out_dir / "03_palette.png"
    save_png(outline_numbered, outline_path)
    save_png(ref.astype(np.uint8), ref_path)
    save_png(palette_img, palette_path)
    
    pdf_path = out_dir / "color_by_numbers_kit.pdf"
    
    # Llama a la nueva función de exportación
    export_pdf(
        [outline_path, palette_path, ref_path],
        pdf_path,
        dpi=args.dpi,
        orientation=args.orientation,
        page_size_name=args.page_size,
        page_wmm=args.page_width_mm,
        page_hmm=args.page_height_mm
    )

    # CSV paleta
    csv_path = out_dir / "palette.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["number","R","G","B","HEX"])
        for i, rgbc in enumerate(centroids_ordered, start=1):
            R,G,B = [int(v) for v in rgbc]; wr.writerow([i,R,G,B,to_hex((R,G,B))])

    print("✅ Listo")
    print(f"- Outline:     {outline_path}")
    print(f"- Paleta:      {palette_path}")
    print(f"- Referencia:  {ref_path}")
    print(f"- PDF:         {pdf_path}")
    print(f"- CSV:         {csv_path}")

if __name__ == "__main__":
    main()