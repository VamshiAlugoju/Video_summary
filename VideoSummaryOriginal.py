 
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
import warnings
import torch
import os
import concurrent.futures
from functools import partial
from collections import defaultdict, deque
from io import BytesIO
from PIL import Image
import logging
 
# Suppress noisy VP8 decoder logs
logging.getLogger("aiortc.codecs.vpx").setLevel(logging.ERROR)
 
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
        
        # User-specific session data
        self.user_sessions = {}  # user_id -> session data
        
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

    def get_user_session_data(self, user_id):
        """Get or initialize user-specific session data"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'caption_buffer': [],
                'frame_count': 0,
                'prev_results': None,
                'prev_depth_map': None,
                'prev_caption': None,
                'track_history': defaultdict(list),
                'track_plate_map': {}
            }
        return self.user_sessions[user_id]

    def cleanup_user_session(self, user_id):
        """Clean up user session data when user disconnects"""
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
            print(f"Cleaned up session data for user: {user_id}")

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

    async def process_frames_ml_only(self, frames, user_session):
        """Process frames with ML operations only - accumulate captions for specific user"""
        user_id = user_session.sid
        print(f"Processing {len(frames)} frames for user {user_id}...")
        
        # Get user-specific session data
        session_data = self.get_user_session_data(user_id)
        
        try:
            for i, frame_data in enumerate(frames):
                frame = frame_data.to_ndarray(format="bgr24")
                if frame is None:
                    continue
                
                session_data['frame_count'] += 1
                
                # Process every nth frame or first frame with parallel ML operations
                if (session_data['frame_count'] % self.DETECTION_INTERVAL == 0 or 
                    session_data['prev_results'] is None):
                    
                    # Run ML operations in parallel
                    detection_result, depth_map, caption = await self.run_ml_operations_parallel(frame)
                    
                    # Only align caption if detection succeeded
                    if detection_result is not None:
                        caption = await align_caption_with_yolo(
                            caption, detection_result, frame, self.image_to_text
                        )
                    
                    # Update session data
                    session_data['prev_results'] = detection_result
                    session_data['prev_depth_map'] = depth_map
                    session_data['prev_caption'] = caption
                else:
                    # Use previous results for intermediate frames
                    caption = session_data['prev_caption'] or "No caption available"
                
                # Continue with existing annotation logic
                if (session_data['prev_results'] is not None and 
                    session_data['prev_depth_map'] is not None):
                    
                    combined_text = await asyncio.to_thread(
                        annotate_frame,
                        frame,
                        session_data['prev_results'],
                        self.model,
                        session_data['prev_depth_map'],
                        session_data['track_history'],
                        conf_thresh=self.CONFIDENCE_THRESHOLD,
                        max_history=self.MAX_TRACK_HISTORY
                    )
                    
                    merged_caption = f"{caption} | {combined_text}" if combined_text else caption
                    
                    # Detect vehicle number plates in a thread
                    vehicle_plate_info = await asyncio.to_thread(
                        self.detect_vehicle_number_with_labels,
                        frame,
                        session_data['prev_results'],
                        session_data['track_plate_map']
                    )
                    
                    vehicle_plate_info = await asyncio.to_thread(
                        format_vehicle_id, vehicle_plate_info
                    )
                    
                    if vehicle_plate_info:
                        merged_caption += " | Vehicles detected: " + ", ".join(vehicle_plate_info)
                    
                    # Add to user-specific caption buffer
                    session_data['caption_buffer'].append(merged_caption)
                
                # Send progress update every 10 frames
                if i % 10 == 0:
                    progress = (i + 1) / len(frames) * 100
                    print(f"Processing progress: {progress:.1f}% for user {user_id}")
            
            return {
                "status": "ML processing completed",
                "total_captions": len(session_data['caption_buffer']),
                "user_id": user_id
            }
                
        except Exception as e:
            print(f"Error processing ML operations for user {user_id}: {e}")
            return {"error": f"Error processing video: {str(e)}", "user_id": user_id}

    async def generate_summary_and_audio(self, user_session, options={"target_language": "eng"}):
        """Generate summary and audio for accumulated captions of specific user"""
        user_id = user_session.sid
        session_data = self.get_user_session_data(user_id)
        
        try:
            if len(session_data['caption_buffer']) > 2:
                print(f"Generating 5-second summary for user {user_id}...")
                
                # Generate summary for the entire 5-second period
                summary = await asyncio.to_thread(
                    generate_summary, session_data['caption_buffer'][2:], self.chain
                )
                print(f"Summary generated for user {user_id}: {summary}")
                
                # Generate audio in a thread
                audio_file = await asyncio.to_thread(
                    request_translation, summary, options.get("target_language", "eng")
                )
                print(f"Audio generated for user {user_id}")
                
                # Clear buffer after processing
                processed_captions = len(session_data['caption_buffer'])
                session_data['caption_buffer'] = []
                
                return {
                    "summary": summary,
                    "audio_file_path": audio_file,
                    "total_captions": processed_captions,
                    "buffer_cleared": True,
                    "user_id": user_id
                }
            else:
                print(f"Insufficient data for summary generation for user {user_id} (need more than 2 captions)")
                processed_captions = len(session_data['caption_buffer'])
                session_data['caption_buffer'] = []
                
                return {
                    "summary": "Insufficient data for summary generation",
                    "audio_file_path": None,
                    "total_captions": processed_captions,
                    "buffer_cleared": True,
                    "user_id": user_id
                }
                
        except Exception as e:
            print(f"Error generating summary for user {user_id}: {e}")
            return {"error": f"Error generating summary: {str(e)}", "user_id": user_id}

    async def process_video_frames(self, frames, user_session, options={"target_language": "eng"}, is_final_chunk=True):
        """Main entry point - process frames with user isolation"""
        user_id = user_session.sid
        
        try:
            # First, run ML operations and accumulate captions for this specific user
            ml_result = await self.process_frames_ml_only(frames, user_session)
            
            if "error" in ml_result:
                return ml_result
            
            # Only generate summary and audio if this is the final chunk (every 5 seconds)
            if is_final_chunk:
                summary_result = await self.generate_summary_and_audio(user_session, options)
                
                # Combine results
                return {
                    **ml_result,
                    **summary_result,
                    "is_final_chunk": True
                }
            else:
                # Intermediate chunk - just return ML processing status
                return {
                    **ml_result,
                    "summary": None,
                    "audio_file_path": None,
                    "buffer_cleared": False,
                    "is_final_chunk": False,
                    "status": f"Captions buffered for user {user_id}, waiting for 5-second completion"
                }
                
        except Exception as e:
            print(f"Error processing video frames for user {user_id}: {e}")
            return {"error": f"Error processing video: {str(e)}", "user_id": user_id}

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

    def get_user_session_stats(self, user_id):
        """Get statistics for specific user session"""
        if user_id in self.user_sessions:
            session_data = self.user_sessions[user_id]
            return {
                'user_id': user_id,
                'caption_buffer_size': len(session_data['caption_buffer']),
                'frame_count': session_data['frame_count'],
                'has_prev_results': session_data['prev_results'] is not None
            }
        return None

    def get_all_session_stats(self):
        """Get statistics for all active user sessions"""
        return {user_id: self.get_user_session_stats(user_id) 
                for user_id in self.user_sessions.keys()}

    def cleanup(self):
        """Clean up thread pool on shutdown"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
            print("Thread pool cleaned up")




video_summary_server = VideoSummaryServer()


__all__ = ["video_summary_server"]


