import socket
import pickle
import numpy as np
import cv2
import os
from typing import List, Tuple, Dict, Any
import tempfile
import struct
import time

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from paddleocr._pipelines.doc_preprocessor import DocPreprocessor

# -----------------------------
# Global models (load once)
# -----------------------------
# docTR OCR model
doctr_model = ocr_predictor(pretrained=True)
doctr_model = doctr_model.to('cuda')

# PaddleOCR preprocessor (orientation + unwarping)
doc_preprocessor = DocPreprocessor(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False
)

def _unwarp_fullframe(image_bgr: np.ndarray) -> np.ndarray:
    """Run PaddleOCR preprocessor (orientation+unwarp). Fallback to original."""
    try:
        results = doc_preprocessor.predict(image_bgr)
        for r in results:
            if "output_img" in r and r["output_img"] is not None:
                return r["output_img"]
    except Exception:
        pass
    return image_bgr

def run_fullframe_ocr(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """
    Run OCR (Paddle preproc + docTR) on the full image.
    Returns a list of word dicts:
      { 'bbox': [x1,y1,x2,y2], 'poly': [[x,y]x4] or None, 'text': str, 'conf': float }
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    # 1) Orientation/Unwarp
    proc = _unwarp_fullframe(image_bgr)

    # 2) docTR expects image path or numpy->DocumentFile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, proc)
        doc = DocumentFile.from_images(tmp.name)

    result = doctr_model(doc)
    h, w = proc.shape[:2]

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
                    "poly": None,  # docTR word geometry here as rect; poly not required for mapping
                    "text": txt,
                    "conf": float(conf),
                })
    #print(words)
    try:
        os.remove(tmp.name)  # clean temp file if possible (doc is a list of paths)
    except Exception:
        pass
    return words


def recv_all(sock, length):
    data = b""
    while len(data) < length:
        packet = sock.recv(length - len(data))
        if not packet:
            return None
        data += packet
    return data

def start_server(host="0.0.0.0", port=7008):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen(5)
        print(f"[OCR Server] Listening on {host}:{port}...")
        while True:
            conn, addr = s.accept()
            with conn:
                # Receive length
                length_data = recv_all(conn, 4)
                if not length_data:
                    continue
                length = struct.unpack("!I", length_data)[0]

                # Receive full data
                data = recv_all(conn, length)
                frame_bgr = pickle.loads(data)

                # Run OCR
                #number_detection_time = time.time()
                words = run_fullframe_ocr(frame_bgr)
                #print(f"number detection time = {time.time()-number_detection_time}")


                # Send length-prefixed response
                resp_data = pickle.dumps(words)
                conn.sendall(struct.pack("!I", len(resp_data)) + resp_data)

if __name__ == "__main__":
    start_server()