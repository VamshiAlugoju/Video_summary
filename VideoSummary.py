


import asyncio
from aiohttp import web
from aiortc.mediastreams import MediaStreamTrack 
from aiortc.mediastreams import AudioStreamTrack
import socketio
from aiortc import RTCPeerConnection, RTCRtpSender, RTCSessionDescription , MediaStreamTrack  
from aiortc.contrib.media import MediaPlayer
import time
import json
import torch
import os
from collections import defaultdict
import wave
import av 
from aiortc.mediastreams import AudioFrame
from random import randint
import shutil
import sounddevice as sd



import warnings
from transformers import logging
import base64
import numpy as np
import cv2

from constants import random_summary , lang_map



# from utils_video_summary import (
#     get_depth_map, detect_and_track, annotate_frame,
#     generate_caption_frame, draw_caption_with_background,
#     generate_summary, align_caption_with_yolo, run_ocr_remote,
#     match_ocr_to_plates, find_vehicle_for_plate, validate_indian_plate,
# )

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from ultralytics import YOLO


 

class VideoSummaryServer:
    def __init__(self):
        # Session management
       
        
        # Initialize models
        self.setup_models()
        print("✅ Video Summary Server initialized")
    
    def setup_models(self):
        """Initialize all the AI models"""
        warnings.filterwarnings("ignore")
        logging.set_verbosity_error()
        logging.get_logger('ultralytics').setLevel(logging.WARNING)
        
     
        
        # self.device = 0 if torch.cuda.is_available() else -1
        # print(f"🔧 Using {'GPU' if self.device == 0 else 'CPU'}")
        
        
        # os.environ["GOOGLE_API_KEY"] = "AIzaSyAombfT-eTn6ZP3a5q02D2Ie1UsJf05l9s"
        
       
        # self.model = YOLO("best.pt")
        # print("✅ YOLO model loaded for Object Detection")
        
        
        # self.plate_detector_anpr = YOLO('license_plate_detector.pt')
        # print("✅ License plate detector loaded")
        
        
        # self.image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        # self.depth_model = DPTForDepthEstimation.from_pretrained(
        #     "Intel/dpt-hybrid-midas", 
        #     low_cpu_mem_usage=True
        # ).to(self.device)
        # print("✅ MIDAS model loaded for Depth Estimation")
        
       
        # self.image_to_text = pipeline(
        #     "image-to-text", 
        #     model="nlpconnect/vit-gpt2-image-captioning", 
        #     device=self.device
        # )
        # print("✅ VIT-GPT2 Model Loaded for Image Captioning")
        
       
        # self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
        
        
        # prompt_template = """
        # You are given a sequence of frame-wise image captions from a video. Each line corresponds to a single frame and includes:
        # - A brief scene caption
        # - Object information in this format:
        #   "Caption | ID1: object_name - depth units, pos(vertical-horizontal); ID2: object_name - depth units, pos(vertical-horizontal); ..."
        # - Optional vehicle number plate data in this format:
        #   "| Vehicles detected: IDx: vehicle_type - PLATENUMBER, IDy: vehicle_type - PLATENUMBER"

        # Guidelines:
        # 1. The object IDs (e.g., ID1, ID2) remain consistent across frames, helping track how the same object moves or changes.
        # 2. Each object includes:
        #    - class name (e.g., car, truck)
        #    - depth value (in units, but use only for relative positioning)
        #    - vertical-horizontal position (e.g., bottom-left)
        #    - optionally, a vehicle number plate (Indian format only).
        # 3. Your tasks:
        #    - Create a unified and fluent summary of the video, describing the overall scene as it evolves over time.
        #    - Integrate relative depth and spatial movement naturally:
        #      * Use expressions like "moved closer", "remained far in the background", "approached from the right".
        #      * Do not include raw numeric depth values – convert them into relative terms like "close", "mid-distance", "far".
        #    - If vehicle number plates are present:
        #      * Include them in the summary, but only once for each unique plate in the whole video.
        #      * Link each valid plate to the type of vehicle and its movements in the scene.
        # 4. Focus only on tracked objects with depth and position info. Skip irrelevant or incomplete entries.
        # 5. Do not output a per-object bullet list. Write a smooth, visually descriptive narrative.
        # 6. If no valid plate numbers are detected, omit plate-related details entirely.
        # 7. The narrative should be as if explaining to someone who cannot see – coherent, vivid, and continuous.

        # Here is the input:
        # {text}
        # UNIFIED SUMMARY:
        # """
        
        # prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        # self.chain = load_summarize_chain(self.llm, chain_type="stuff", prompt=prompt)
        # print("✅ GEMINI2.0 Model Loaded for Summarization")
        
       
        self.DETECTION_INTERVAL = 10
        self.CONFIDENCE_THRESHOLD = 0.4
        self.MAX_TRACK_HISTORY = 30
     
    

    def detect_vehicle_number_with_labels(self, frame_bgr, yolo_results, track_plate_map):
        """Detect vehicle number plates with labels"""
        # Get vehicle bboxes + labels + track IDs
        # boxes = yolo_results.boxes.xyxy.cpu().numpy()
        # classes = yolo_results.boxes.cls.int().cpu().numpy()
        # track_ids = (
        #     yolo_results.boxes.id.int().cpu().numpy()
        #     if yolo_results.boxes.id is not None
        #     else [None] * len(boxes)
        # )

        # vehicle_info = []  # [(track_id, label, bbox)]
        # for box, cls_id, tid in zip(boxes, classes, track_ids):
        #     label = yolo_results.names[int(cls_id)]
        #     if label.lower() in ("car", "truck", "bus", "motorcycle"):
        #         vehicle_info.append((tid, label, [int(b) for b in box]))

        # if not vehicle_info:
        #     return []  # No vehicles

        # vehicle_boxes = [v[2] for v in vehicle_info]

        # # Detect license plates
        # plate_results = self.plate_detector_anpr(frame_bgr)[0]
        # plate_boxes = [
        #     [int(x1), int(y1), int(x2), int(y2)]
        #     for x1, y1, x2, y2, conf, *_ in plate_results.boxes.data.tolist()
        #     if conf > 0.3
        # ]
        # if not plate_boxes:
        #     return []

        # # OCR full frame via remote server
        # try:
        #     ocr_words = run_ocr_remote(frame_bgr)
        # except Exception as e:
        #     print(f"⚠ OCR server error: {e}")
        #     return []

        # # Match OCR → plates
        # plate_texts = match_ocr_to_plates(
        #     ocr_words,
        #     plate_boxes,
        #     frame_bgr=frame_bgr,
        #     per_plate_ocr=False
        # )

        # # Map plate to vehicle + track ID with validation
        # vehicle_plate_pairs = []
        # for pt in plate_texts:
        #     vb = find_vehicle_for_plate(pt["plate_bbox"], vehicle_boxes)
        #     if vb:
        #         for tid, label, vbox in vehicle_info:
        #             if vbox == vb and pt["text"]:
        #                 # Validate and normalize plate
        #                 validation = validate_indian_plate(pt["text"])
        #                 if validation["is_valid"]:
        #                     formatted_plate = validation["formatted"]

        #                     # Keep stable plate mapping for each track_id
        #                     if tid not in track_plate_map:
        #                         track_plate_map[tid] = formatted_plate

        #                     vehicle_plate_pairs.append(
        #                         f"ID{tid}: {label} - {track_plate_map[tid]}"
        #                     )

        # return vehicle_plate_pairs
    
    # async def process_video_frames(self , frames , sid):
    #     print("Processing video frames")
    #     loop = asyncio.get_running_loop()
    #     loop.run_in_executor(None, time.sleep, 18)
    #     print("Processing video frames done , 18 sec is completed")
    #     summary = "some text"
    #     return summary

    
    async def process_video_frames(self, frames,sid , options = {"target_language" : "eng"}):
        """Process accumulated video frames and generate summary"""
        try:
            captions = []
            
            frame_count = 0
            prev_results = None
            prev_depth_map = None
            track_history = defaultdict(list)
            track_plate_map = {}
            target_language = options["target_language"]

            print(f"🎬 Processing {len(frames)} frames...")
            await asyncio.sleep(5)
            
            
            # for i, frame in enumerate(frames):
                
            #     if frame is None:
            #         continue
                
            #     frame_count += 1
                
            #     # Process every nth frame or first frame
            #     if True or prev_results is None:
            #           temp= 1
            #         # prev_results = detect_and_track(frame, self.model)
            #         # prev_depth_map = get_depth_map(
            #         #     frame, self.depth_model, self.image_processor, self.device
            #         # )
            #         # caption = generate_caption_frame(frame, self.image_to_text)
            #         # caption = align_caption_with_yolo(
            #         #     caption, prev_results, frame, self.image_to_text
            #         # )
                
            #     # Annotate frame (get combined text)
            #     # _, combined_text = annotate_frame(
            #     #     frame, prev_results, self.model, prev_depth_map, 
            #     #     track_history, conf_thresh=self.CONFIDENCE_THRESHOLD, 
            #     #     max_history=self.MAX_TRACK_HISTORY
            #     # )
                
            #     # merged_caption = f"{caption} | {combined_text}" if combined_text else caption
                
            #     # Detect vehicle number plates
            #     # vehicle_plate_info = self.detect_vehicle_number_with_labels(
            #     #     frame, prev_results, track_plate_map
            #     # )
                
            #     # if vehicle_plate_info:
            #     #     merged_caption += " | Vehicles detected: " + ", ".join(vehicle_plate_info)
                
            #     # captions.append(merged_caption)
                

            
            print("🤖 Generating summary...")
            
            # Generate summary using your existing function
            # summary = generate_summary(captions[2:], self.chain)
            summary = random_summary
            # copy the ./sample_15.wav to ./audios. change name to sample_15_{random}.wav
            file_path = "./sample_15.wav"
            new_file_path = "./audios/sample_15_" + str(randint(1, 100)) + ".wav"
            shutil.copy(file_path, new_file_path)
            audio_file_path = new_file_path
            print("✅ Summary generated successfully")
            # return summary 
            return {
                "summary" : summary,
                "audio_file_path" : audio_file_path
            }
            
        except Exception as e:
            print(f"⚠ Error processing video frames: {e}")
            return f"Error processing video: {str(e)}"
    

video_summary_server = VideoSummaryServer()

__all__ = ["video_summary_server"]