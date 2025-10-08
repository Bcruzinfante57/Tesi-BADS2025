import numpy as np
import cv2
from PIL import Image
from rembg import remove 
from pathlib import Path
import time
import sys
import traceback

# --- Configuración del Modelo de REMBG (CRÍTICO: Manejo de Importación de new_session) ---

MODEL_NAME_MAIN = "isnet-general-use" 
MODEL_NAME_FALLBACK = "u2net" 

session_main = None
session_fallback = None
session_ready = False # Estado global para saber si podemos hacer doble segmentación

try:
    # 1. Intenta importar la forma moderna para usar modelos específicos (IS-NET)
    from rembg.sessions import new_session
    session_main = new_session(model_name=MODEL_NAME_MAIN)
    session_fallback = new_session(model_name=MODEL_NAME_FALLBACK)
    session_ready = True
    print(f"✅ REMBG: Modelos '{MODEL_NAME_MAIN}' y '{MODEL_NAME_FALLBACK}' cargados para doble segmentación.")

except ImportError:
    # 2. Fallback si la versión de REMBG es antigua y no tiene 'new_session'
    print("❌ WARNING: 'new_session' no está disponible. Usando segmentación simple (u2net) por defecto.")
    session_ready = False
    
# --- Constantes de Ajuste ---
CUSTOM_PADDING = 0.20 # Padding ajustado al 20%.

# --- Funciones Auxiliares ---

def crop_object_with_padding(pil_img_rgba, padding_ratio=CUSTOM_PADDING): 
    """
    Recorta el objeto principal utilizando el bounding box de TODOS los píxeles no transparentes.
    """
    
    if pil_img_rgba.mode != 'RGBA':
        pil_img_rgba = pil_img_rgba.convert("RGBA")

    alpha_channel_np = np.array(pil_img_rgba.split()[-1])
    
    # Encontrar todos los píxeles no transparentes (alpha > 0)
    coords = np.where(alpha_channel_np > 0)
    
    if coords[0].size == 0:
        print("Warning: No píxeles opacos encontrados en el canal Alpha. Recorte no aplicado.")
        # Retorna la imagen RGBA, convertida a fondo blanco sin recorte.
        cropped_img_rgb = Image.new("RGB", pil_img_rgba.size, (255, 255, 255))
        cropped_img_rgb.paste(pil_img_rgba, mask=pil_img_rgba.split()[-1])
        return cropped_img_rgb
        
    # Calcular el Bounding Box mínimo.
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    w = x_max - x_min
    h = y_max - y_min
    
    # Calcular Padding
    pad_x = int(w * padding_ratio) 
    pad_y = int(h * padding_ratio)
    
    # Aplicar padding con límites de la imagen
    x0 = max(0, x_min - pad_x)
    y0 = max(0, y_min - pad_y)
    x1 = min(pil_img_rgba.size[0], x_max + pad_x) 
    y1 = min(pil_img_rgba.size[1], y_max + pad_y) 

    # Aplicar recorte a la imagen RGBA
    cropped_rgba = pil_img_rgba.crop((x0, y0, x1, y1))
    
    # Post-Procesamiento: Convertir a fondo blanco
    cropped_img_rgb = Image.new("RGB", cropped_rgba.size, (255, 255, 255))
    alpha_channel_cropped = cropped_rgba.split()[-1]
    cropped_img_rgb.paste(cropped_rgba, mask=alpha_channel_cropped)

    return cropped_img_rgb

# --- PRIMARY FLOW FUNCTION ---

def process_and_crop_pipeline(pil_img, padding_ratio=CUSTOM_PADDING):
    """
    Aplica el flujo completo: 
    1. Si es posible: Doble segmentación con fusión de máscaras (IS-NET + U2NET).
    2. Si falla: Segmentación simple (U2NET por defecto).
    3. Recorte por Bounding Box con padding.
    """
    
    pil_img_rgb = pil_img.convert("RGB") 

    if session_ready:
        # --- MODO 1: DOBLE SEGMENTACIÓN (Mejor precisión) ---
        print("    -> Ejecutando MODO AVANZADO: Doble Segmentación y Fusión de Máscaras.")

        # 1. SEGMENTACIÓN PRINCIPAL (IS-NET)
        try:
            start_time = time.time()
            print(f"    -> Ejecutando REMBG [1/2] - Modelo: '{MODEL_NAME_MAIN}' (Enfoque objeto sólido)")
            mask1_rgba = remove(pil_img_rgb, session=session_main, alpha_matting=True)
            alpha1_np = np.array(mask1_rgba.split()[-1])
            print(f"    -> Segmentación 1 completada en {time.time() - start_time:.2f} s.")
        except Exception as e:
            print(f"Error en Segmentación 1 ({MODEL_NAME_MAIN}): {e}. Usando máscara vacía.")
            alpha1_np = np.zeros(pil_img_rgb.size[::-1], dtype=np.uint8)

        # 2. SEGMENTACIÓN SECUNDARIA (U2NET + Alpha Matting)
        try:
            start_time = time.time()
            print(f"    -> Ejecutando REMBG [2/2] - Modelo: '{MODEL_NAME_FALLBACK}' (Enfoque bordes finos)")
            mask2_rgba = remove(pil_img_rgb, session=session_fallback, alpha_matting=True)
            alpha2_np = np.array(mask2_rgba.split()[-1])
            print(f"    -> Segmentación 2 completada en {time.time() - start_time:.2f} s.")
        except Exception as e:
            print(f"Error en Segmentación 2 ({MODEL_NAME_FALLBACK}): {e}. Usando máscara vacía.")
            alpha2_np = np.zeros(pil_img_rgb.size[::-1], dtype=np.uint8)
        
        # 3. FUSIÓN DE MÁSCARAS (OR lógico)
        alpha_fusion_np = np.maximum(alpha1_np, alpha2_np)
        print("    -> Máscaras fusionadas (OR lógico) para preservar el puente.")

        # 4. Crear imagen RGBA final
        rgb_channels = list(pil_img_rgb.split())
        alpha_fusion_img = Image.fromarray(alpha_fusion_np) 
        clean_img_rgba = Image.merge('RGBA', rgb_channels + [alpha_fusion_img])

    else:
        # --- MODO 2: FALLBACK (Si 'new_session' falla o no existe) ---
        print("    -> Ejecutando MODO FALLBACK: Segmentación simple (u2net) con alpha matting agresivo.")
        try:
            start_time = time.time()
            clean_img_rgba = remove(pil_img_rgb, alpha_matting=True)
            print(f"    -> REMBG completado en {time.time() - start_time:.2f} segundos.")
        except Exception as e:
            print(f"Error durante la ejecución de rembg: {e}. Volviendo a la imagen original.")
            traceback.print_exc()
            return pil_img_rgb

    # 5. CROP: Aplicar recorte basado en el Bounding Box
    cropped_img_rgb = crop_object_with_padding(
        clean_img_rgba, 
        padding_ratio=padding_ratio
    )
    
    return cropped_img_rgb

# --- FUNCIÓN DE PROCESAMIENTO DE CARPETAS ---

def process_all_images_in_folder(input_folder_path, output_folder_name="processed_images", **kwargs):
    """
    Itera sobre todos los archivos de imagen en una carpeta de entrada, aplica el
    pipeline de procesamiento y guarda el resultado en una carpeta de salida.
    """
    input_folder = Path(input_folder_path)
    if not input_folder.is_dir():
        print(f"❌ Error: La carpeta de entrada no se encontró en: {input_folder}")
        return

    output_folder = input_folder / output_folder_name
    output_folder.mkdir(exist_ok=True)
    
    image_paths = sorted([p for p in input_folder.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
    
    if not image_paths:
        print(f"⚠️ No se encontraron imágenes en la carpeta: {input_folder}")
        return

    print(f"\n--- Procesando imágenes en: {input_folder.name} ({len(image_paths)} archivos) ---")
    
    for img_path in image_paths:
        try:
            print(f"\n  → Procesando {img_path.name}...")
            pil_img = Image.open(img_path)
            
            # Aplicar la pipeline de procesamiento
            processed_img = process_and_crop_pipeline(pil_img, padding_ratio=kwargs.get('padding_ratio', CUSTOM_PADDING))
            
            # Guardar la imagen procesada
            output_path = output_folder / img_path.name
            processed_img.save(output_path)
            print(f"  ✓ Guardado en {output_path}")

        except Exception as e:
            print(f"  ❌ Fallo crítico al procesar {img_path.name}: {e}. Skipping.")
            traceback.print_exc()


    print("\n--- Procesamiento de carpeta completado ---")


if __name__ == "__main__":
    
    # --- BOILERPLATE DE CARGA DE DATOS ORIGINAL (MANTENIDO) ---
    brands = ["test_crop"]
    brand_folders = {"test_crop": "test_crop"}
    
    for brand in brands:
        folder = Path(brand_folders.get(brand, "")).resolve()
        if not folder.exists():
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except OSError:
                print(f"❌ Cannot create folder for {brand}: {folder}. Exiting simulation.")
                continue 

    # --- EJECUTAR EL PROCESAMIENTO DE IMÁGENES ---
    
    TEST_FOLDER = "test_crop" 
    
    print(f"\n🚨 EJECUTANDO PIPELINE V24: MODO HÍBRIDO + MÁSCARA DE FUSIÓN 🚨")
    
    process_all_images_in_folder(
        input_folder_path=TEST_FOLDER, 
        output_folder_name="processed_output_v24_hybrid_final", 
        padding_ratio=CUSTOM_PADDING
    )