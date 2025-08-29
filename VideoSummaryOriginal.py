 
import asyncio
from aiohttp import web
from aiortc.mediastreams import MediaStreamTrack
import socketio
from aiortc import RTCPeerConnection, RTCSessionDescription
import time
import json
import base64
import cv2
import numpy as np
import time
import threading
import warnings
import torch
import os
from collections import defaultdict
from io import BytesIO
from PIL import Image

# Import your existing modules
from transformers import DPTImageProcessor, DPTForDepthEstimation, pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from transformers import logging
from ultralytics import YOLO

from utils_video_summary import (
    get_depth_map, detect_and_track, annotate_frame,
    generate_caption_frame, draw_caption_with_background,
    generate_summary, align_caption_with_yolo, run_ocr_remote,
    match_ocr_to_plates, find_vehicle_for_plate, validate_indian_plate,
    request_translation, play_audio_file, run_fullframe_ocr,
)


def format_vehicle_id(input_list):
    output_list = []
    for item in input_list:
        if "-" in item:
            prefix, code = item.split("-", 1)  # Split into prefix and code
            code_with_spaces = " ".join(code.strip())  # Add spaces between characters
            formatted = f"{prefix.strip()} - {code_with_spaces}"
            output_list.append(formatted)
        else:
            output_list.append(item)  # In case there's no '-'
    return output_list
 
 
class VideoSummaryServer:
    def __init__(self):
        # Session management
        self.sessions = {}  # websocket -> session data
        
        # Language mapping
        self.lang_map = {
            "arb": "Arabic", "ben": "Bengali", "cat": "Catalan", "ces": "Czech", "cmn": "Mandarin Chinese",
            "cym": "Welsh", "dan": "Danish", "deu": "German", "eng": "English", "est": "Estonian",
            "fin": "Finnish", "fra": "French", "hin": "Hindi", "ind": "Indonesian", "ita": "Italian",
            "jpn": "Japanese", "kor": "Korean", "mlt": "Maltese", "nld": "Dutch", "pes": "Persian",
            "pol": "Polish", "por": "Portuguese", "ron": "Romanian", "rus": "Russian", "slk": "Slovak",
            "spa": "Spanish", "swe": "Swedish", "swh": "Swahili", "tel": "Telugu", "tgl": "Tagalog",
            "tha": "Thai", "tur": "Turkish", "ukr": "Ukrainian", "urd": "Urdu", "uzn": "Uzbek",
            "vie": "Vietnamese"
        }
        
        # Initialize models
        self.setup_models()
        
        # Create thread pool for parallel ML operations
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        print("Video Summary Server initialized")
    
    def setup_models(self):
        """Initialize all the AI models"""
        warnings.filterwarnings("ignore")
        logging.set_verbosity_error()
        logging.get_logger('ultralytics').setLevel(logging.WARNING)
        
        # Set up device
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"Using {'GPU' if self.device == 0 else 'CPU'}")
        
        # Set up Google API key
        os.environ["GOOGLE_API_KEY"] = "AIzaSyAombfT-eTn6ZP3a5q02D2Ie1UsJf05l9s"
        
        # Load YOLO model
        self.model = YOLO("best.pt")
        print("YOLO model loaded for Object Detection")
        
        # Load ANPR models
        self.plate_detector_anpr = YOLO('license_plate_detector.pt')
        print("License plate detector loaded")
        
        # Load depth estimation model
        self.image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        self.depth_model = DPTForDepthEstimation.from_pretrained(
            "Intel/dpt-hybrid-midas", 
            low_cpu_mem_usage=True
        ).to(self.device)
        print("MIDAS model loaded for Depth Estimation")
        
        # Load captioning model
        self.image_to_text = pipeline(
            "image-to-text", 
            model="nlpconnect/vit-gpt2-image-captioning", 
            device=self.device
        )
        print("VIT-GPT2 Model Loaded for Image Captioning")
        
        # Load LLM for summarization
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
        
        # Set up summarization prompt
        prompt_template = """
        You are given a sequence of frame-wise image captions from a video. Each line corresponds to a single frame and includes:
        - A brief scene caption
        - Object information in this format:
        "Caption | ID1: object_name - depth units, pos(vertical-horizontal); ID2: object_name - depth units, pos(vertical-horizontal); ..."
        - Optional vehicle number plate data in this format:
        "| Vehicles detected: IDx: vehicle_type - PLATE NUMBER, IDy: vehicle_type - PLATE NUMBER"

        IMPORTANT RULE ABOUT NUMBER PLATES:
        - Number plates are provided with spaces between characters (e.g., "T S 1 3 E P 1 0 2 6"). 
        - Do NOT merge, compress, or normalize them into a single string like "TS13EP1026". 
        - Always keep the plate format exactly as provided with spaces. 
        - Reason: The final summary will be converted to speech, and merging makes the TTS pronounce the plate incorrectly or as garbage audio.

        Guidelines:
        1. The object IDs (e.g., ID1, ID2) remain consistent across frames, helping track how the same object moves or changes.
        2. Each object includes:
        - class name (e.g., car, truck)
        - depth value (in units, but use only for relative positioning)
        - vertical-horizontal position (e.g., bottom-left)
        - optionally, a vehicle number plate (Indian format only, written with spaces as given).
        3. Your tasks:
        - Create a unified and fluent summary of the video, describing the overall scene as it evolves over time.
        - Integrate relative depth and spatial movement naturally:
            * Use expressions like "moved closer", "remained far in the background", "approached from the right".
            * Do not include raw numeric depth values — convert them into relative terms like "close", "mid-distance", "far".
        - If vehicle number plates are present:
            * Include them in the summary, but only once for each unique plate in the whole video.
            * Link each valid plate to the type of vehicle and its movements in the scene.
            * Always preserve the plate exactly as given (with spaces).
        4. Focus only on tracked objects with depth and position info. Skip irrelevant or incomplete entries.
        5. Do not output a per-object bullet list. Write a smooth, visually descriptive narrative.
        6. If no valid plate numbers are detected, omit plate-related details entirely.
        7. The narrative should be as if explaining to someone who cannot see — coherent, vivid, and continuous.

        Here is the input:
        {text}
        UNIFIED SUMMARY:
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        self.chain = load_summarize_chain(self.llm, chain_type="stuff", prompt=prompt)
        print("GEMINI2.0 Model Loaded for Summarization")
        
        self.DETECTION_INTERVAL = 1
        self.CONFIDENCE_THRESHOLD = 0.4
        self.MAX_TRACK_HISTORY = 30

    async def run_ml_operations_parallel(self, frame):
        """Run the three ML operations in parallel"""
        loop = asyncio.get_event_loop()
        
        # Create partial functions for thread execution
        detect_task = partial(detect_and_track, frame, self.model)
        depth_task = partial(get_depth_map, frame, self.depth_model, self.image_processor, self.device)
        caption_task = partial(generate_caption_frame, frame, self.image_to_text)
        
        try:
            # Run all three operations concurrently
            results = await asyncio.gather(
                loop.run_in_executor(self.thread_pool, detect_task),
                loop.run_in_executor(self.thread_pool, depth_task),
                loop.run_in_executor(self.thread_pool, caption_task),
                return_exceptions=True
            )
            
            # Handle any exceptions
            detection_result, depth_map, caption = results
            
            if isinstance(detection_result, Exception):
                print(f"Detection failed: {detection_result}")
                detection_result = None
            if isinstance(depth_map, Exception):
                print(f"Depth estimation failed: {depth_map}")
                depth_map = None
            if isinstance(caption, Exception):
                print(f"Captioning failed: {caption}")
                caption = "Failed to generate caption"
            
            return detection_result, depth_map, caption
            
        except Exception as e:
            print(f"Parallel ML operations failed: {e}")
            return None, None, "Failed to generate caption"
    
    def base64_to_frame(self, base64_string):
        """Convert base64 string to OpenCV frame"""
        try:
            # Remove the data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64 to bytes
            img_bytes = base64.b64decode(base64_string)
            
            # Convert to numpy array
            nparr = np.frombuffer(img_bytes, np.uint8)
            
            # Decode to OpenCV image
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return frame
        except Exception as e:
            print(f"Error converting base64 to frame: {e}")
            return None
    
    def detect_vehicle_number_with_labels(self, frame_bgr, yolo_results, track_plate_map):
        """Detect vehicle number plates with labels"""
        # Get vehicle bboxes + labels + track IDs
        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        classes = yolo_results.boxes.cls.int().cpu().numpy()
        track_ids = (
            yolo_results.boxes.id.int().cpu().numpy()
            if yolo_results.boxes.id is not None
            else [None] * len(boxes)
        )

        vehicle_info = []  # [(track_id, label, bbox)]
        for box, cls_id, tid in zip(boxes, classes, track_ids):
            label = yolo_results.names[int(cls_id)]
            if label.lower() in ("car", "truck", "bus", "motorcycle", "autorickshaw"):
                vehicle_info.append((tid, label, [int(b) for b in box]))

        if not vehicle_info:
            return []  # No vehicles

        vehicle_boxes = [v[2] for v in vehicle_info]

        # Detect license plates
        plate_results = self.plate_detector_anpr(frame_bgr)[0]
        plate_boxes = [
            [int(x1), int(y1), int(x2), int(y2)]
            for x1, y1, x2, y2, conf, *_ in plate_results.boxes.data.tolist()
            if conf > 0.3
        ]
        if not plate_boxes:
            return []

        # OCR full frame via remote server
        try:
            ocr_words = run_fullframe_ocr(frame_bgr)
        except Exception as e:
            print(f"OCR server error: {e}")
            return []

        # Match OCR → plates
        plate_texts = match_ocr_to_plates(
            ocr_words,
            plate_boxes,
            frame_bgr=frame_bgr,
            per_plate_ocr=False
        )

        # Map plate to vehicle + track ID with validation
        vehicle_plate_pairs = []
        for pt in plate_texts:
            vb = find_vehicle_for_plate(pt["plate_bbox"], vehicle_boxes)
            if vb:
                for tid, label, vbox in vehicle_info:
                    if vbox == vb and pt["text"]:
                        # Validate and normalize plate
                        validation = validate_indian_plate(pt["text"])
                        if validation["is_valid"]:
                            formatted_plate = validation["formatted"]

                            # Keep stable plate mapping for each track_id
                            if tid not in track_plate_map:
                                track_plate_map[tid] = formatted_plate

                            vehicle_plate_pairs.append(
                                f"ID{tid}: {label} - {track_plate_map[tid]}"
                            )
        return vehicle_plate_pairs
    
    async def generate_and_play_audio(self, summary, language, websocket):
        """Generate audio from summary and play it"""
        try:
            # Notify client that audio generation started
            await websocket.send(json.dumps({
                'type': 'processing_status',
                'message': f'Generating audio in {self.lang_map.get(language, language)}...'
            }))
            
            print(f"Generating audio for language: {language} ({self.lang_map.get(language, language)})")
            
            # Generate audio using translation service
            audio_file = request_translation(summary, language)
            
            if audio_file and os.path.exists(audio_file):
                # Notify client that audio is playing
                await websocket.send(json.dumps({
                    'type': 'audio_playing',
                    'message': f'Playing audio in {self.lang_map.get(language, language)}'
                }))
                
                print(f"Playing audio file: {audio_file}")
                
                # Play audio (this will block until audio finishes)
                play_audio_file(audio_file)
                
                # Notify client that audio completed
                await websocket.send(json.dumps({
                    'type': 'audio_complete',
                    'message': 'Audio playback completed'
                }))
                
                print("Audio playback completed")
                
                # Clean up audio file
                try:
                    #os.remove(audio_file)
                    print(f"Cleaned up audio file: {audio_file}")
                except:
                    pass
            else:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': 'Failed to generate audio file'
                }))
                print("Failed to generate audio file")
        
        except Exception as e:
            print(f"Error in audio generation: {e}")
            await websocket.send(json.dumps({
                'type': 'error',
                'message': f'Audio generation failed: {str(e)}'
            }))

    async def run_ml_operations_parallel(self, frame):
        """Run the three ML operations in parallel"""
        loop = asyncio.get_event_loop()
        
        # Create partial functions for thread execution
        detect_task = partial(detect_and_track, frame, self.model)
        depth_task = partial(get_depth_map, frame, self.depth_model, self.image_processor, self.device)
        caption_task = partial(generate_caption_frame, frame, self.image_to_text)
        
        try:
            # Run all three operations concurrently
            results = await asyncio.gather(
                loop.run_in_executor(self.thread_pool, detect_task),
                loop.run_in_executor(self.thread_pool, depth_task),
                loop.run_in_executor(self.thread_pool, caption_task),
                return_exceptions=True
            )
            
            # Handle any exceptions
            detection_result, depth_map, caption = results
            
            if isinstance(detection_result, Exception):
                print(f"Detection failed: {detection_result}")
                detection_result = None
            if isinstance(depth_map, Exception):
                print(f"Depth estimation failed: {depth_map}")
                depth_map = None
            if isinstance(caption, Exception):
                print(f"Captioning failed: {caption}")
                caption = "Failed to generate caption"
            
            return detection_result, depth_map, caption
            
        except Exception as e:
            print(f"Parallel ML operations failed: {e}")
            return None, None, "Failed to generate caption"

    async def process_video_frames(self, frames):
        """Process accumulated video frames with parallel ML operations"""
        print(f"Processing {len(frames)} frames...")
        
        try:
            captions = []
            frame_count = 0
            prev_results = None
            prev_depth_map = None
            prev_caption = None
            track_history = defaultdict(list)
            track_plate_map = {}
            
            for i, frame_data in enumerate(frames):
                frame = frame_data.to_ndarray(format="bgr24")
                if frame is None:
                    continue
                
                frame_count += 1
                
                # Process every nth frame or first frame with parallel ML operations
                if frame_count % self.DETECTION_INTERVAL == 0 or prev_results is None:
                    # Run ML operations in parallel
                    prev_results, prev_depth_map, caption = await self.run_ml_operations_parallel(frame)
                    
                    # Only align caption if detection succeeded
                    if prev_results is not None:
                        caption = align_caption_with_yolo(
                            caption, prev_results, frame, self.image_to_text
                        )
                    prev_caption = caption
                else:
                    # Use previous results for intermediate frames
                    caption = prev_caption or "No caption available"
                
                # Continue with existing annotation logic
                if prev_results is not None and prev_depth_map is not None:
                    _, combined_text = annotate_frame(
                        frame, prev_results, self.model, prev_depth_map,
                        track_history, conf_thresh=self.CONFIDENCE_THRESHOLD,
                        max_history=self.MAX_TRACK_HISTORY
                    )
                    
                    merged_caption = f"{caption} | {combined_text}" if combined_text else caption
                    
                    # Detect vehicle number plates
                    vehicle_plate_info = self.detect_vehicle_number_with_labels(
                        frame, prev_results, track_plate_map
                    )
                    vehicle_plate_info = format_vehicle_id(vehicle_plate_info)
                    
                    if vehicle_plate_info:
                        merged_caption += " | Vehicles detected: " + ", ".join(vehicle_plate_info)
                    
                    captions.append(merged_caption)
                
                # Send progress update every 10 frames
                if i % 10 == 0:
                    progress = (i + 1) / len(frames) * 100
                    print(f"Processing progress: {progress:.1f}%")
            
            print("Generating summary...")
            
            # Generate summary using your existing function
            if len(captions) > 2:
                summary = generate_summary(captions[2:], self.chain)
                print(summary)
            else:
                summary = "Insufficient data for summary generation"
                
            print("Summary generated successfully")


            # Generate audio
            #audio_file = request_translation(summary, "eng")
            print("Audio generated")

            return summary
            
        except Exception as e:
            print(f"Error processing video frames: {e}")
            return f"Error processing video: {str(e)}"
    
    def cleanup(self):
        """Clean up thread pool on shutdown"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
            print("Thread pool cleaned up")


video_summary_server = VideoSummaryServer()

__all__ = ["video_summary_server"]


