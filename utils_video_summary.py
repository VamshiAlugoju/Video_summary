# utils.py

import cv2
import numpy as np
import torch
import textwrap
import json
import socket
from PIL import Image
from collections import defaultdict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pydub import AudioSegment
from pydub.playback import play
from threading import Thread
import re 
import pickle 
import struct
import os
from typing import List, Tuple, Dict, Any
import asyncio

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# -----------------------------
# Global models (load once)
# -----------------------------
# docTR OCR model
doctr_model = ocr_predictor(pretrained=True)
doctr_model = doctr_model.to('cuda')


# ---------- DEPTH + DETECTION + TRACKING ----------

def get_depth_map(frame, model, image_processor, device):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(frame_rgb)
    inputs = image_processor(images=image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image_pil.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    depth_map = prediction.squeeze().cpu().numpy()
    midas_depth_inv = 1.0 / (depth_map + 1e-6)
    return midas_depth_inv * (6 / 0.0050)

def get_color_by_id(track_id):
    np.random.seed(track_id % 100)
    return tuple(int(x) for x in np.random.randint(0, 255, size=3))

def detect_and_track(frame, model):
    return model.track(frame, persist=True)[0]

def annotate_frame(frame, results, model, depth_map, track_history=None, conf_thresh=0.4, max_history=30, image_mode=False):
    annotated = frame.copy()
    depth_texts = []
    frame_height, frame_width = frame.shape[:2]

    if results.boxes and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.int().cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.int().cpu().numpy()

        for idx, (box, track_id, conf, cls_id) in enumerate(zip(boxes, ids, confs, classes)):
            if conf < conf_thresh:
                continue

            x1, y1, x2, y2 = map(int, box)
            x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2

            # No tracking logic if image_mode
            if not image_mode and track_history is not None:
                track = track_history[track_id]
                track.append((x_center, y_center))
                if len(track) > max_history:
                    track.pop(0)

            depth_crop = depth_map[y1:y2, x1:x2]
            dominant_depth = None
            if depth_crop.size > 0:
                hist, bin_edges = np.histogram(depth_crop.flatten(), bins=50)
                if np.max(hist) > 0:
                    idx = np.argmax(hist)
                    dominant_depth = (bin_edges[idx] + bin_edges[idx + 1]) / 2.0

            horiz = "left" if x_center < frame_width / 3 else "right" if x_center > 2 * frame_width / 3 else "center"
            vert = "top" if y_center < frame_height / 3 else "bottom" if y_center > 2 * frame_height / 3 else "center"
            position = f"{vert}-{horiz}"

            label = model.names[cls_id]
            obj_id_str = f"ID{track_id}" if not image_mode else f"OBJ{idx}"
            text = f"{obj_id_str}: {label} - {dominant_depth:.2f} units, pos({position})" if dominant_depth else f"{obj_id_str}: {label} - N/A, pos({position})"
            depth_texts.append(text)

            color = get_color_by_id(track_id if not image_mode else idx)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if not image_mode and track_history is not None:
                points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [points], isClosed=False, color=color, thickness=2)

    return annotated, ", ".join(depth_texts)

# ---------- CAPTIONING + OVERLAY ----------

def generate_caption_frame(frame, caption_pipeline):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(frame_rgb)
    result = caption_pipeline(image_pil)
    return result[0]['generated_text'].strip()

async def align_caption_with_yolo(global_caption, yolo_results, frame, image_to_text):
    """
    Align global caption with YOLO detected objects using object-level captions.

    Args:
        global_caption (str): Frame-level caption (ViT-GPT2).
        yolo_results (Results): YOLO results object from Ultralytics.
        frame (ndarray): Full video frame.
        image_to_text: HuggingFace pipeline for image captioning.

    Returns:
        str: Updated caption aligned with YOLO detections.
    """
    corrected_map = {}

    # --- Convert YOLO results into iterable detections ---
    try:
        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        labels = yolo_results.boxes.cls.cpu().numpy().astype(int)
    except Exception as e:
        print("[align_caption_with_yolo] YOLO results parsing failed:", e)
        return global_caption

    names = yolo_results.names  # class id → label

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        yolo_label = names[labels[i]]
        #print(f"label:{yolo_label}")

        # Crop YOLO object
        obj_crop = frame[y1:y2, x1:x2]

        # 🔹 Use your generate_caption_frame for the crop
        obj_caption = ""
        try:
            obj_caption = await asyncio.to_thread(generate_caption_frame, obj_crop, image_to_text)
            obj_caption = obj_caption.lower()
        except Exception as e:
            print(f"[align_caption_with_yolo] Caption failed for {yolo_label}: {e}")
        #print(f"object_caption: {obj_caption}")

        # Find if caption says car/bus/truck/etc
        match = re.search(r"(car|bus|truck|van|vehicle|motorcycle)", obj_caption)
        if match:
            predicted = match.group(1)
            corrected_map[predicted] = yolo_label  # replace ViT guess with YOLO truth

    # --- Update global caption with YOLO-aligned labels ---
    updated_caption = global_caption
    for predicted, yolo_label in corrected_map.items():
        updated_caption = re.sub(predicted, yolo_label, updated_caption, flags=re.IGNORECASE)

    return updated_caption

def draw_caption_with_background(frame, caption, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, font_thickness=1):
    h, w, _ = frame.shape
    caption_lines = textwrap.wrap(caption, width=50)
    line_height = 25
    box_height = line_height * len(caption_lines) + 10
    box_y_start = h - box_height
    cv2.rectangle(frame, (0, box_y_start), (w, h), (0, 0, 0), thickness=-1)
    y = box_y_start + line_height
    for line in caption_lines:
        x = 10
        cv2.putText(frame, line, (x, y), font, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
        y += line_height
    return frame

# ---------- SUMMARIZATION ----------

def generate_summary(captions, summarization_chain):
    text = " ".join([f"Frame {i+1}: {caption}." for i, caption in enumerate(captions)])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = [Document(page_content=t) for t in text_splitter.split_text(text)]
    return summarization_chain.run(docs)

# ---------- TRANSLATION + AUDIO ----------

def select_tgt_language():
    lang_map = {
        "arb": "Arabic", "ben": "Bengali", "cat": "Catalan", "ces": "Czech", "cmn": "Mandarin Chinese",
        "cym": "Welsh", "dan": "Danish", "deu": "German", "eng": "English", "est": "Estonian",
        "fin": "Finnish", "fra": "French", "hin": "Hindi", "ind": "Indonesian", "ita": "Italian",
        "jpn": "Japanese", "kor": "Korean", "mlt": "Maltese", "nld": "Dutch", "pes": "Persian",
        "pol": "Polish", "por": "Portuguese", "ron": "Romanian", "rus": "Russian", "slk": "Slovak",
        "spa": "Spanish", "swe": "Swedish", "swh": "Swahili", "tel": "Telugu", "tgl": "Tagalog",
        "tha": "Thai", "tur": "Turkish", "ukr": "Ukrainian", "urd": "Urdu", "uzn": "Uzbek",
        "vie": "Vietnamese"
    }

    lang_list = list(enumerate(sorted(lang_map.items(), key=lambda x: x[1]), 1))
    print("Select a target language:\n")
    for i, (code, name) in lang_list:
        print(f"{i}. {name} ({code})")

    try:
        choice = int(input("\nEnter the number corresponding to your language choice: "))
        if 1 <= choice <= len(lang_list):
            selected_code = lang_list[choice - 1][1][0]
            selected_name = lang_list[choice - 1][1][1]
            print(f"\n✅ You selected: {selected_name} ({selected_code})")
        else:
            raise ValueError
    except ValueError:
        print("❌ Invalid input.")
        return "eng"
    return selected_code

def request_translation(text, tgt_lang, host='localhost', port=51008):
    request = json.dumps({"text": text, "tgt_lang": tgt_lang})
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(request.encode())
        chunks = []
        while True:
            chunk = s.recv(1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b''.join(chunks).decode().strip()

def play_audio_file(audio_file):
    play(AudioSegment.from_wav(audio_file))

def play_audio_file_async(audio_file):
    """Plays audio file in a separate thread."""
    def _play():
        sound = AudioSegment.from_wav(audio_file)
        play(sound)

    t = Thread(target=_play)
    t.start()
    return t  # return thread if caller wants to monitor

def generate_image_summary(caption, object_info, chain):
    full_text = f"{caption} | {object_info}"
    doc = Document(page_content=full_text)
    return chain.run([doc])


#OCR recognition:

def run_ocr_remote(frame_bgr, host="localhost", port=7008):
    data = pickle.dumps(frame_bgr)
    length = struct.pack("!I", len(data))  # 4-byte big-endian length

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(length + data)

        # Receive length of response
        resp_len_data = s.recv(4)
        resp_len = struct.unpack("!I", resp_len_data)[0]

        # Receive full response
        resp_data = b""
        while len(resp_data) < resp_len:
            packet = s.recv(4096)
            if not packet:
                break
            resp_data += packet

    return pickle.loads(resp_data)

# util.py
import os
import cv2
import numpy as np
import string
import tempfile
from typing import List, Tuple, Dict, Any

import re

# More comprehensive character correction mappings
dict_char_to_int = {
    'O': '0', 'I': '1', 'S': '5', 'G': '6', 'Z': '2', 'B': '8',
    'D': '0', 'Q': '0', 'L': '1', 'T': '7', 'A': '4'  # Additional common OCR errors
}

dict_int_to_char = {
    '0': 'O', '1': 'I', '5': 'S', '6': 'G', '2': 'Z', '8': 'B',
    '4': 'A', '7': 'T'  # Additional mappings
}

# Indian plate patterns with proper regex
INDIAN_PLATE_PATTERNS = [
    re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{1,4}$'),    # Standard: AA##AAA#### (most common)
    re.compile(r'^[A-Z]{2}[0-9]{1}[A-Z]{1,3}[0-9]{1,4}$'),     # Single district digit: AA#AAA####
    re.compile(r'^[A-Z]{3}[0-9]{1,2}[A-Z]{1,2}[0-9]{1,4}$'),   # 3-letter state code (rare)
]

def extract_plate_segments(text: str) -> dict:
    """
    Intelligently extract segments from Indian license plate text.
    Returns dict with 'state', 'district', 'series', 'number' segments.
    """
    if not text or len(text) < 6:
        return None
    
    # Try to identify the pattern by finding digit groups
    digits_positions = [i for i, c in enumerate(text) if c.isdigit()]
    letters_positions = [i for i, c in enumerate(text) if c.isalpha()]
    
    if len(digits_positions) < 2 or len(letters_positions) < 3:
        return None
    
    # Find the first group of digits (district code)
    first_digit_start = digits_positions[0]
    first_digit_end = first_digit_start
    while (first_digit_end + 1 < len(text) and 
           text[first_digit_end + 1].isdigit() and 
           first_digit_end - first_digit_start < 2):  # Max 2 digits for district
        first_digit_end += 1
    
    # Find the last group of digits (unique number)
    last_digit_end = digits_positions[-1]
    last_digit_start = last_digit_end
    while (last_digit_start - 1 >= 0 and 
           text[last_digit_start - 1].isdigit() and 
           last_digit_end - last_digit_start < 4):  # Max 4 digits for number
        last_digit_start -= 1
    
    try:
        segments = {
            'state': text[:first_digit_start],           # Letters before first digit group
            'district': text[first_digit_start:first_digit_end+1],  # First digit group
            'series': text[first_digit_end+1:last_digit_start],     # Letters between digit groups
            'number': text[last_digit_start:last_digit_end+1]       # Last digit group
        }
        
        # Validate segment lengths
        if (2 <= len(segments['state']) <= 3 and 
            1 <= len(segments['district']) <= 2 and 
            1 <= len(segments['series']) <= 3 and 
            1 <= len(segments['number']) <= 4):
            return segments
    except:
        pass
    
    return None

def correct_segment_characters(segment: str, segment_type: str) -> str:
    """
    Correct characters in a segment based on expected type (letters or digits).
    """
    corrected = []
    for char in segment:
        if segment_type in ['state', 'series']:  # Should be letters
            if char.isdigit():
                corrected.append(dict_int_to_char.get(char, char))
            else:
                corrected.append(char)
        elif segment_type in ['district', 'number']:  # Should be digits
            if char.isalpha():
                corrected.append(dict_char_to_int.get(char, char))
            else:
                corrected.append(char)
        else:
            corrected.append(char)
    
    return ''.join(corrected)

def format_license(text: str) -> str:
    """
    Normalize OCR plate text to Indian format with intelligent segmentation.
    """
    if not text:
        return ""
    
    # Clean input
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    if not cleaned:
        return ""
    
    # Check if already valid
    if any(pattern.match(cleaned) for pattern in INDIAN_PLATE_PATTERNS):
        return cleaned
    
    # Try to extract and correct segments
    segments = extract_plate_segments(cleaned)
    
    if segments:
        # Correct each segment
        corrected_state = correct_segment_characters(segments['state'], 'state')
        corrected_district = correct_segment_characters(segments['district'], 'district')
        corrected_series = correct_segment_characters(segments['series'], 'series')
        corrected_number = correct_segment_characters(segments['number'], 'number')
        
        # Reconstruct plate
        corrected_plate = corrected_state + corrected_district + corrected_series + corrected_number
        
        # Validate the corrected plate
        if any(pattern.match(corrected_plate) for pattern in INDIAN_PLATE_PATTERNS):
            return corrected_plate
    
    # Fallback: try trimming extra characters from start/end
    for trim_start in range(min(3, len(cleaned))):  # Try removing up to 2 chars from start
        for trim_end in range(min(3, len(cleaned) - trim_start)):  # Try removing up to 2 chars from end
            if trim_start == 0 and trim_end == 0:
                continue
            
            trimmed = cleaned[trim_start:len(cleaned)-trim_end if trim_end > 0 else len(cleaned)]
            
            if len(trimmed) < 6:  # Too short
                continue
            
            # Try segment extraction on trimmed version
            segments = extract_plate_segments(trimmed)
            if segments:
                corrected_state = correct_segment_characters(segments['state'], 'state')
                corrected_district = correct_segment_characters(segments['district'], 'district')
                corrected_series = correct_segment_characters(segments['series'], 'series')
                corrected_number = correct_segment_characters(segments['number'], 'number')
                
                corrected_plate = corrected_state + corrected_district + corrected_series + corrected_number
                
                if any(pattern.match(corrected_plate) for pattern in INDIAN_PLATE_PATTERNS):
                    return corrected_plate
    
    # Last resort: return cleaned text (might not be valid)
    return cleaned

def license_complies_format(text: str) -> bool:
    """
    Validate against Indian plate patterns after formatting.
    """
    if not text:
        return False
    
    formatted = format_license(text)
    return any(pattern.match(formatted) for pattern in INDIAN_PLATE_PATTERNS)

def validate_indian_plate(plate: str) -> dict:
    """
    Comprehensive validation with detailed feedback.
    """
    formatted = format_license(plate)
    is_valid = any(pattern.match(formatted) for pattern in INDIAN_PLATE_PATTERNS)
    
    segments = extract_plate_segments(formatted) if formatted else None
    
    return {
        'original': plate,
        'formatted': formatted,
        'is_valid': is_valid,
        'segments': segments,
        'length': len(formatted) if formatted else 0
    }



# -----------------------------
# Geometry helpers
# -----------------------------
def _center(bbox: List[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def _ioa(inner: List[int], outer: List[int]) -> float:
    """Intersection over area of 'inner' box (useful for containment scoring)."""
    x1, y1, x2, y2 = inner
    X1, Y1, X2, Y2 = outer
    inter_x1 = max(x1, X1)
    inter_y1 = max(y1, Y1)
    inter_x2 = min(x2, X2)
    inter_y2 = min(y2, Y2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    inner_area = max(1, (x2 - x1) * (y2 - y1))
    return inter / inner_area

# -----------------------------
# Map OCR -> plate boxes
# -----------------------------
def _center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def _ioa(boxA, boxB):
    # Intersection over area of A
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    if boxA_area == 0:
        return 0
    return inter_area / boxA_area

def run_fullframe_ocr(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """
    Run OCR (docTR only) on the full image.
    Returns a list of word dicts:
      { 'bbox': [x1,y1,x2,y2], 'poly': None, 'text': str, 'conf': float }
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    # Save to temp file for docTR input
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, image_bgr)
        doc = DocumentFile.from_images(tmp.name)

    result = doctr_model(doc)
    h, w = image_bgr.shape[:2]

    words = []
    try:
        page = result.pages[0]
    except Exception:
        return words

    # Extract words with absolute pixel bboxes
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                # word.geometry is ((xmin,ymin),(xmax,ymax)) normalized [0,1]
                (x_min, y_min), (x_max, y_max) = word.geometry
                x1 = int(x_min * w)
                y1 = int(y_min * h)
                x2 = int(x_max * w)
                y2 = int(y_max * h)
                txt = (word.value or "").strip()
                if not txt:
                    continue
                conf = getattr(word, "confidence", 1.0)
                words.append({
                    "bbox": [x1, y1, x2, y2],
                    "poly": None,  # docTR only provides rectangular geometry
                    "text": txt,
                    "conf": float(conf),
                })
    try:
        os.remove(tmp.name)  # clean up temp file
    except Exception:
        pass
    return words

def match_ocr_to_plates(ocr_words, plate_boxes, frame_bgr=None, per_plate_ocr=False):
    """
    Robustly match OCR tokens to each plate bbox and merge multi-line plates.
    Returns: [{ "plate_bbox": [x1,y1,x2,y2], "text": "<merged>" }, ...]
    If per_plate_ocr=True and frame_bgr is provided, runs OCR on plate crops
    instead of full-frame tokens (usually better).
    """
    def center_inside(b, x1, y1, x2, y2):
        cx = (b[0] + b[2]) / 2.0
        cy = (b[1] + b[3]) / 2.0
        return (x1 <= cx <= x2) and (y1 <= cy <= y2)

    def group_into_lines(words, plate_h):
        # Group by y-center with adaptive threshold
        if not words:
            return []
        words = sorted(words, key=lambda w: ((w['bbox'][1] + w['bbox'][3]) / 2.0,
                                             (w['bbox'][0] + w['bbox'][2]) / 2.0))
        lines = []
        # 18% of plate height (>= 6 px)
        thr = max(6, int(0.18 * plate_h))
        for w in words:
            y_c = (w['bbox'][1] + w['bbox'][3]) // 2
            if not lines:
                lines.append({'y_vals': [y_c], 'words': [w], 'y_mean': y_c})
                continue
            # attach to nearest line by y-distance
            diffs = [abs(y_c - ln['y_mean']) for ln in lines]
            idx = min(range(len(lines)), key=lambda i: diffs[i])
            if diffs[idx] <= thr:
                ln = lines[idx]
                ln['words'].append(w)
                ln['y_vals'].append(y_c)
                ln['y_mean'] = sum(ln['y_vals']) / len(ln['y_vals'])
            else:
                lines.append({'y_vals': [y_c], 'words': [w], 'y_mean': y_c})
        # order lines top→bottom, and words left→right in each line
        for ln in lines:
            ln['words'].sort(key=lambda w: (w['bbox'][0] + w['bbox'][2]) / 2.0)
        lines.sort(key=lambda ln: ln['y_mean'])
        return lines

    results = []
    for pb in plate_boxes:
        x1, y1, x2, y2 = map(int, pb)
        pw = max(1, x2 - x1)
        ph = max(1, y2 - y1)

        if per_plate_ocr and frame_bgr is not None:
            # Better: OCR on the plate crop itself (less noise)
            crop = frame_bgr[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
            #words_in_plate = run_ocr_remote(crop)  # returns local coords in the crop
            words_in_plate = run_fullframe_ocr(crop)  # returns local coords in the crop

            # Shift crop coords to full-frame if you later need absolute positions
            for w in words_in_plate:
                bx = w['bbox']
                w['bbox'] = [bx[0] + x1, bx[1] + y1, bx[2] + x1, bx[3] + y1]
        else:
            # Use full-frame OCR tokens but keep only those whose center lies inside the plate box
            words_in_plate = [w for w in ocr_words if center_inside(w['bbox'], x1, y1, x2, y2)]

        if not words_in_plate:
            continue

        # 1) Group into lines robustly
        lines = group_into_lines(words_in_plate, plate_h=ph)

        # 2) Build per-line strings
        line_texts = [''.join(w['text'] for w in ln['words']) for ln in lines if ln['words']]

        # 3) Make candidates: top→bottom, bottom→top
        candidates = []
        if line_texts:
            candidates.append(''.join(line_texts))            # top→bottom
            if len(line_texts) >= 2:
                candidates.append(''.join(line_texts[::-1]))  # bottom→top

        # 4) Validate candidates; pick the first that passes
        chosen = None
        for cand in candidates:
            v = validate_indian_plate(cand)
            if v['is_valid']:
                chosen = v['formatted']
                break

        # 5) If nothing validated, try segment-aware normalization on each candidate
        if chosen is None:
            best = None
            for cand in candidates:
                normalized = format_license(cand)
                vv = validate_indian_plate(normalized)
                if vv['is_valid']:
                    best = vv['formatted']
                    break
            chosen = best if best is not None else (candidates[0] if candidates else "")

        if chosen:
            results.append({"plate_bbox": pb, "text": chosen})

    return results

# -----------------------------
# Vehicle mapping
# -----------------------------
def find_vehicle_for_plate(plate_bbox: List[int], vehicle_boxes: List[List[int]]) -> List[int]:
    """
    Return the vehicle bbox that contains the plate (first full containment match).
    If none fully contains it, return the one with max IoU/IoA.
    """
    px1, py1, px2, py2 = plate_bbox
    best_idx = -1
    best_score = -1.0

    for i, vb in enumerate(vehicle_boxes):
        vx1, vy1, vx2, vy2 = vb
        # full containment
        if px1 >= vx1 and py1 >= vy1 and px2 <= vx2 and py2 <= vy2:
            return vb

        # otherwise use IoA of plate in vehicle
        score = _ioa([px1, py1, px2, py2], [vx1, vy1, vx2, vy2])
        if score > best_score:
            best_score = score
            best_idx = i

    return vehicle_boxes[best_idx] if best_idx >= 0 else None

# -----------------------------
# CSV writer (optional)
# -----------------------------
def write_csv(results_per_image: List[Dict[str, Any]], output_path: str) -> None:
    """
    results_per_image: list of {
        'filename': str,
        'items': [ {'vehicle_bbox': [..] or None, 'plate_bbox': [..], 'text': str or None, 'score': float} ... ]
    }
    """
    with open(output_path, "w") as f:
        f.write("filename,vehicle_bbox,plate_bbox,plate_text,score\n")
        for row in results_per_image:
            fn = row.get("filename", "")
            for it in row.get("items", []):
                vb = it.get("vehicle_bbox")
                pb = it.get("plate_bbox")
                txt = it.get("text")
                sc = it.get("score", 0.0)
                f.write(f"{fn},{vb},{pb},{txt},{sc:.3f}\n")