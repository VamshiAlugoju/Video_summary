random_summary = "As I sat at my desk with the faint hum of my laptop fan competing against the muffled sounds of traffic outside, I couldn’t help but reflect on the peculiar nature of ambition, how it drives us to chase goals that often seem distant and intimidating, and yet, in the quiet moments between the chaos, I began to wonder if ambition is less about the finish line we imagine and more about the endless process of showing up each day, confronting fears that disguise themselves as excuses, and building a kind of resilience that no textbook or mentor can fully explain, because resilience, I realized, is forged in the tiny decisions we repeat over and over — like choosing to keep writing when the words feel clumsy, or continuing to exercise when the mirror doesn’t immediately reward the effort, or persisting with a project when the people around us don’t entirely understand its value — and the longer I thought about it, the more it struck me that perhaps the greatest measure of success is not the recognition we eventually receive but the stubborn willingness to keep going in the face of uncertainty, to cultivate a rhythm of persistence that turns even our smallest, most fragile actions into proof that fear does not hold the final word."




import asyncio
from aiohttp import web
from aiortc.mediastreams import MediaStreamTrack
import socketio
from aiortc import RTCPeerConnection, RTCSessionDescription
import time
import json
import torch
import os
from collections import defaultdict



import warnings
from transformers import logging
import base64
import numpy as np
import cv2


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




# ----------------------------
# Socket.IO server setup
# ----------------------------
sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
app = web.Application()
sio.attach(app)


video_sessions = {}
 

INTERVAL_SEC = 5
PROCESSING_FPS = 5

#  ------- global video summary server----------
video_summary_server = None 

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
    
    async def process_video_frames(self, frames):
        """Process accumulated video frames and generate summary"""
        try:
            captions = []
            
            frame_count = 0
            prev_results = None
            prev_depth_map = None
            track_history = defaultdict(list)
            track_plate_map = {}
            
            print(f"🎬 Processing {len(frames)} frames...")
            
            for i, frame in enumerate(frames):
                
                if frame is None:
                    continue
                
                frame_count += 1
                
                # Process every nth frame or first frame
                if True or prev_results is None:
                      temp= 1
                    # prev_results = detect_and_track(frame, self.model)
                    # prev_depth_map = get_depth_map(
                    #     frame, self.depth_model, self.image_processor, self.device
                    # )
                    # caption = generate_caption_frame(frame, self.image_to_text)
                    # caption = align_caption_with_yolo(
                    #     caption, prev_results, frame, self.image_to_text
                    # )
                
                # Annotate frame (get combined text)
                # _, combined_text = annotate_frame(
                #     frame, prev_results, self.model, prev_depth_map, 
                #     track_history, conf_thresh=self.CONFIDENCE_THRESHOLD, 
                #     max_history=self.MAX_TRACK_HISTORY
                # )
                
                # merged_caption = f"{caption} | {combined_text}" if combined_text else caption
                
                # Detect vehicle number plates
                # vehicle_plate_info = self.detect_vehicle_number_with_labels(
                #     frame, prev_results, track_plate_map
                # )
                
                # if vehicle_plate_info:
                #     merged_caption += " | Vehicles detected: " + ", ".join(vehicle_plate_info)
                
                # captions.append(merged_caption)
                

            
            print("🤖 Generating summary...")
            
            # Generate summary using your existing function
            # summary = generate_summary(captions[2:], self.chain)
            
            print("✅ Summary generated successfully")
            # return summary
            return ""
            
        except Exception as e:
            print(f"⚠ Error processing video frames: {e}")
            return f"Error processing video: {str(e)}"
    


#    ---------------track-sessions-----------------
 

# ----------------------------
# Track peer connections
# ----------------------------
peers = {}  # sid -> RTCPeerConnection



async def delayed_processing(frames , sid):
    if frames:
        print(f"⏳ {len(frames)} frames collected, processing now..." , video_sessions[sid]['prev_summary_processed_time'])
        # Example processing call
        current_fps = len(frames) / (time.time() - video_sessions[sid]['prev_summary_processed_time'])
        video_sessions[sid]['prev_summary_processed_time'] = time.time()
        print(f"FPS: {current_fps}")
        skip_frames = max(1, int(current_fps / PROCESSING_FPS)) # PROCESSING_FPS is 5
        filtered_frames = []
        for i in range(0, len(frames), skip_frames):
            filtered_frames.append(frames[i])
        print("🚀 Processing frames size :", len(filtered_frames))
        await sio.emit("summary_processing" , {"status" : True} , to=sid)
        summary = await video_summary_server.process_video_frames(filtered_frames)
        
        await sio.emit("summary_end" , {"summary" : random_summary} , to=sid)
        return summary
 
# ----------------------------
# Socket.IO events
# ----------------------------
@sio.event
async def connect(sid, environ):
    print(f"🔌 Socket connected: {sid}")
    video_sessions[sid] = {
    "pc": RTCPeerConnection(),   # the WebRTC peer connection
    "frames": [],                # store aiortc VideoFrame objects
    "is_recording": False,       # optional flag if needed
    "start_time": None,          # when recording started
    "stream_start_time": None,   # WebRTC stream start
    "stream_end_time": None,      # WebRTC stream stop
    "prev_summary_processed_time" : None
    }

@sio.event
async def stop_stream(sid): 
     print(f"❌ Stop streaming from {sid}")
    
     video_sessions[sid]['stream_end_time'] = time.time()
     session = video_sessions.get(sid, None)
     frames = session["frames"]
     print("📊 Stop streaming from", sid, "— total frames:", len(frames), "— total time:", session['stream_end_time'] - session['stream_start_time'])
     asyncio.create_task(delayed_processing(frames, sid))
     if session and session["pc"]:
         await session["pc"].close()

@sio.event
async def disconnect(sid):
    print(f"❌ Socket disconnected: {sid}")
    session = video_sessions.pop(sid, None)
    if session and session["pc"]:
        await session["pc"].close()

@sio.event
async def offer(sid, data):
    print(f"📡 Received offer from {sid}")

    offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
    pc = RTCPeerConnection()
   
    video_sessions[sid]["pc"] = pc

    @pc.on("track")
    def on_track(track: MediaStreamTrack):
        print(f"🎥 Track received from {sid}: {track.kind}")
        video_sessions[sid]['stream_start_time'] = time.time()
        video_sessions[sid]['prev_summary_processed_time'] = time.time()

        if track.kind == "video":
            

            async def log_frames_and_process():
                
                stream_start_time = time.time()  # local to this async function
                try:
                    while True:
                        
                        frame = await track.recv()
                        video_sessions[sid]["frames"].append(frame)
                        
                        time_diff = time.time() - stream_start_time
                        # print("⏳ Time diff:", time_diff, "frames " , frame )
                        if time_diff >= INTERVAL_SEC:
                            frames_to_process = video_sessions[sid]["frames"][:]
                            video_sessions[sid]["frames"] = []
                            asyncio.create_task(delayed_processing(frames_to_process, sid))
                            stream_start_time = time.time()  # reset timer
                except Exception:
                    print(f"Track ended for {sid}, stopping frame logging")

            asyncio.create_task(log_frames_and_process())

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    await sio.emit("answer", {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}, to=sid)


# ----------------------------
# App startup
# ----------------------------
async def on_startup(app):
    global video_summary_server
    video_summary_server = VideoSummaryServer()
    print("✅ VideoSummaryServer initialized after server start")

app.on_startup.append(on_startup)

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    web.run_app(app, port=8080)
