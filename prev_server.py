
# =================== working previous server file for reference ==================== 





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


from audio_player import AudioPlayer, PlayerStreamTrackPlayer

from user_manager import UserManager , User


ROOT = os.path.dirname(__file__)

async def create_audio_track(file_path , on_ended):
    player = AudioPlayer(file_path , on_ended=on_ended)  # decode=True by default
    return player.audio

 



# ----------------------------
# Socket.IO server setup
# ----------------------------
sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
app = web.Application()
sio.attach(app)


video_sessions = {}
 

INTERVAL_SEC = 5
PROCESSING_FPS = 3

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
    
    # async def process_video_frames(self , frames , sid):
    #     print("Processing video frames")
    #     loop = asyncio.get_running_loop()
    #     loop.run_in_executor(None, time.sleep, 18)
    #     print("Processing video frames done , 18 sec is completed")
    #     summary = "some text"
    #     return summary

    
    async def process_video_frames(self, frames,sid):
        """Process accumulated video frames and generate summary"""
        try:
            captions = []
            
            frame_count = 0
            prev_results = None
            prev_depth_map = None
            track_history = defaultdict(list)
            track_plate_map = {}
            target_language = video_sessions[sid]['target_language']

            await asyncio.sleep(5)
            
            print(f"🎬 Processing {len(frames)} frames...")
            
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
            summary = ""
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
    


#    ---------------track-sessions-----------------
 

# ----------------------------
# Track peer connections
# ----------------------------
peers = {}  # sid -> RTCPeerConnection

files = ['sample_1.wav' , 'sample_2.wav' , 'sample_3.wav' , 'sample_4.wav']
curr_index = 0
async def delayed_processing(frames , sid):
     global curr_index
     await asyncio.sleep(10)
     if curr_index >= len(files):
          print("no more files")
          return
     curr_path = os.path.join(ROOT, f"audios/{files[curr_index]}")
     curr_index += 1
     audio_track = await  create_audio_track(curr_path)
     pc = video_sessions[sid]['pc']
     audio_sender = None
     for sender in pc.getSenders():
          if sender.track and sender.track.kind == "audio":
               audio_sender = sender
               break
     print("replacing track")
     audio_sender.replaceTrack(audio_track)

     
     

    # if frames:
    #     print(f"⏳ {len(frames)} frames collected, processing now..." , video_sessions[sid]['prev_summary_processed_time'])
    #     # Example processing call
    #     current_fps = len(frames) / (time.time() - video_sessions[sid]['prev_summary_processed_time'])
    #     video_sessions[sid]['prev_summary_processed_time'] = time.time()
    #     print(f"FPS: {current_fps}")
    #     skip_frames = max(1, int(current_fps / PROCESSING_FPS)) # PROCESSING_FPS is 5
    #     filtered_frames = []
    #     for i in range(0, len(frames), skip_frames):
    #         filtered_frames.append(frames[i])
    #     print("🚀 Processing frames size :", len(filtered_frames))
    #     await sio.emit("summary_processing" , {"status" : True} , to=sid)
    #     process_output = await video_summary_server.process_video_frames(filtered_frames ,sid)
    #     summary = process_output["summary"]
    #     audio_file_path = process_output["audio_file_path"]
        
    #     await sio.emit("summary_end" , {"summary" : summary , "audio" : audio_file_path} , to=sid)
    #     return summary

queue_of_tracks = []
no_of_tracks = 1
async def add_audio_track(sid): 
    global no_of_tracks
   
    if(no_of_tracks > 4):
        print("no of tracks exceeded")
        return
    print("adding audio track ", no_of_tracks)
    path = os.path.join(ROOT, "sample_song.wav")
    
    audio_track = create_audio_track(path)  # <-- your file here
    pc = video_sessions[sid]['pc']

    queue_of_tracks.append(audio_track)
 
    audio_sender = None
    for sender in  pc.getSenders():
        if sender.track and sender.track.kind == "audio":
            audio_sender = sender
            break
    curr_playing_track = audio_sender.track
    # check whether the current playing track is completed or not. if there is no track then play the next track.
    # if there is a track and it is not completed, wait for the track to complete and then play the next track;

    no_of_tracks += 1 
 

global is_song 
is_song = False

async def on_ended_closure(sid):
    async def on_ended(file_name):
        try:
            print("ended TRack =------------------------" , file_name, "for socket id" , sid)
            pc : RTCPeerConnection = video_sessions[sid]['pc']
            if(pc is None):
                print("pc is none")
                return
            audio_path = os.path.join(ROOT, "sample_song.wav")
            global is_song
            if(is_song):    
                audio_path = os.path.join(ROOT, "sample_15.wav")
                is_song = False
            else:
                is_song = True
            curr_on_ended =await  on_ended_closure(sid)
            audio_track = await  create_audio_track(audio_path , curr_on_ended)  # <-- your file here
            audio_sender = None
            for sender in  pc.getSenders():
                if sender.track and sender.track.kind == "audio":
                    print("audio sender is " , sender)
                    sender.track.stop()
                    audio_sender = sender
                    break
            print("replacing track with " , audio_track)
            print("audio sender is " , audio_sender)
            audio_sender.replaceTrack(audio_track)
           
            
        except Exception as e:
            print("error in on_ended", e)

    return on_ended
 
 
def on_ended(sid):
    try:
        print("ended TRack =------------------------" , sid)
        initial_track : PlayerStreamTrackPlayer =  video_sessions[sid]['initial_track'] 
        print("initial track is " , initial_track)
        path = os.path.join(ROOT, "sample_15.wav")
        initial_track.set_source(path )
    except Exception as e:
        print("error in on_ended", e)
       
    

    

class WebrtcHandler():
    def __init__(self) -> None:
        print("======initiailsed webrtc handler======")
    
    @staticmethod
    async def offer(sid, data):
        print(f"📡 Received offer from {sid}")

        offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
        pc = RTCPeerConnection()
        path = os.path.join(ROOT, "sample_15.wav")
        # on_ended = await on_ended_closure(sid) 
        # audio_track =await create_audio_track(path , on_ended)  # <-- your file here
        silence_track =  PlayerStreamTrackPlayer(on_ended=on_ended , sid=sid)
        video_sessions[sid]["initial_track"] = silence_track
     
        pc.addTrack(silence_track)

        video_sessions[sid]["pc"] = pc

        @pc.on("connectionstatechange")
        async def on_connection_state_change():
             await WebrtcHandler.handle_connection_state_change(pc)
            


        @pc.on("track")
        async def on_track(track: MediaStreamTrack):
            silence_track.set_source(path)
            print("track received", "and track type is " , track.kind)
            # await WebrtcHandler.handle_track(pc, track , sid)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        await sio.emit("answer", {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}, to=sid)
    
    @staticmethod
    async def handle_connection_state_change(pc):
        print("Connection state changed to:", pc.connectionState)
    
    @staticmethod
    async def handle_track(pc : RTCPeerConnection, track : MediaStreamTrack , sid ):
        print(f"🎥 Track received from {pc}: {track.kind}")
       
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
                        if time_diff >= INTERVAL_SEC:
                            frames_to_process = video_sessions[sid]["frames"][:] 
                            video_sessions[sid]["frames"] = []
                            asyncio.create_task(delayed_processing(frames_to_process, sid))
                            stream_start_time = time.time()  # reset timer
                except Exception:
                    print(f"Track ended for {sid}, stopping frame logging")

            asyncio.create_task(log_frames_and_process())

    
webrtc_handler = WebrtcHandler()


async def close_pc_conn(sid):
    pc_outlink : RTCPeerConnection = video_sessions[sid]['pc_outlink']
    print("closing pc conn")
    await pc_outlink.close()
    print("✅ Old PeerConnection closed")

is_song = False

async def on_ended(sid):
    try:
        print(f"🎬 Track ended for SID: {sid}")

      

        # --- Step 2: Close existing PeerConnection ---
        pc_outlink : RTCPeerConnection = video_sessions[sid]['pc_outlink']
        print(f"🔌 Closing old PeerConnection for SID: {sid}, current state: {pc_outlink.signalingState}")
        pc_outlink.cancel()
        pc_outlink.remove_all_listeners()
        for sender in pc_outlink.getSenders():
           await sender.track.stop()
        
        await close_pc_conn(sid)
        print("reciever tracks stopped")
        await pc_outlink.close()
        print("✅ Old PeerConnection closed")
        global is_song
        file_name = ""
        if is_song:
            file_name = "sample_15.wav"
        else:
            file_name = "sample_song.wav"
            
        is_song = not is_song
        another_player = AudioPlayer(os.path.join(ROOT, file_name) ,on_ended = on_ended , sid = sid)
        # --- Step 3: Create new PeerConnection ---
        pc_outlink = RTCPeerConnection()
        video_sessions[sid]['pc_outlink'] = pc_outlink
        print("🔧 New PeerConnection created and stored in video_sessions")

        # --- Step 4: Add new audio track ---
        pc_outlink.addTrack(another_player.audio)
        print("🎶 Added new audio track to PeerConnection")

        # --- Step 5: Create and set local offer ---
        new_offer = await pc_outlink.createOffer()
        print("📝 Created new offer")
        await pc_outlink.setLocalDescription(new_offer)
        print(f"✅ Local description set with SDP:\n{new_offer.sdp[:200]}...")  # print first 200 chars only

        # --- Step 6: Emit new offer to client ---
        await sio.emit(
            "incoming_offer",
            {
                "sdp": pc_outlink.localDescription.sdp,
                "type": pc_outlink.localDescription.type,
                "replace_connection": True
            },
            to=sid
        )
        print("📡 Emitted incoming_offer to client for renegotiation")

    except Exception as e:
        print(f"⚠️ Error in on_ended for SID {sid}: {e}")
       
 

class handleSocket():
    def __init__(self) -> None:
        print("======initiailsed socket handler======")
    
    @staticmethod
    async def connect(sid):
        print(f"🔌 Socket connected: {sid}")
        user_manager.add_user(sid)
        new_pc = RTCPeerConnection()
      
        path = os.path.join(ROOT, "sample_15.wav")
        player =  AudioPlayer(path , on_ended = on_ended , sid = sid)

        new_pc.addTrack(player.audio)
        self_offfer = await new_pc.createOffer()

        await new_pc.setLocalDescription(self_offfer)
        
        await sio.emit("incoming_offer" , { "sdp": new_pc.localDescription.sdp, "type": new_pc.localDescription.type }, to=sid)
        video_sessions[sid] = {
        "pc": RTCPeerConnection(),   # the WebRTC peer connection
        "frames": [],                # store aiortc VideoFrame objects
        "is_recording": False,       # optional flag if needed
        "start_time": None,          # when recording started
        "stream_start_time": None,   # WebRTC stream start
        "stream_end_time": None,      # WebRTC stream stop
        "prev_summary_processed_time" : None , 
        "target_language" : lang_map["eng"],
        "is_streaming" : False,
        "pc_outlink" : RTCPeerConnection(),
        }
        video_sessions[sid]['pc_outlink'] = new_pc
        
       
        

    @staticmethod
    async def disconnect(sid):
        print(f"❌ Socket disconnected: {sid}")
        session = video_sessions.pop(sid, None)
        if session and session["pc"]:
            await session["pc"].close()
    

    
    @staticmethod
    async def select_target_language(sid , data):
        target_language = data["language"]
        video_sessions[sid]['target_language'] = target_language
        print(f"✅ Target language selected: {target_language}")
     

    
    @staticmethod
    async def offer(sid, data):
        await webrtc_handler.offer(sid, data)

    
    @staticmethod
    async def stop_stream(sid): 
     print(f"❌ Stop streaming from {sid}")
    
     video_sessions[sid]['stream_end_time'] = time.time()
     session = video_sessions.get(sid, None)
     frames = session["frames"]
     print("📊 Stop streaming from", sid, "— total frames:", len(frames), "— total time:", session['stream_end_time'] - session['stream_start_time'])
 
     if session and session["pc"]:
         await session["pc"].close()
    
    @staticmethod
    async def renegotiation_answer(sid, data):
        pc = video_sessions.get(sid, {}).get("pc")
        if not pc:
            print(f"❌ No peer connection for {sid}")
            return

        answer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
        await pc.setRemoteDescription(answer)
        print(f"✅ Renegotiation completed for {sid}")

    @staticmethod
    async def incoming_answer(sid ,data):
        pc_outlink : RTCPeerConnection = video_sessions.get(sid, {}).get("pc_outlink")
        if not pc_outlink:
            print(f"❌ No peer connection for {sid}")
            return
        
        answer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
        print("current_state  " , pc_outlink.signalingState )
        await pc_outlink.setRemoteDescription(answer)
         
        async def on_connection_state_change():
            print("🔌 Connection state changed:", pc_outlink.connectionState)
            if pc_outlink.connectionState == "connected":
                # schedule close asynchronously
                print("connection state change")
                 
                 

        pc_outlink.on("connectionstatechange", on_connection_state_change)
        video_sessions[sid]['pc_outlink'] = pc_outlink



socket_handler = handleSocket()

# ----------------------------
# Socket.IO events
# ----------------------------
@sio.event
async def connect(sid, environ):
   await socket_handler.connect(sid)

@sio.event
async def disconnect(sid):
    await socket_handler.disconnect(sid)
    

@sio.event
async def stop_stream(sid): 
    await socket_handler.stop_stream(sid)

@sio.event
async def select_target_language(sid , data):
    await socket_handler.select_target_language(sid , data)
    

@sio.event
async def offer(sid, data):
    await socket_handler.offer(sid, data)

@sio.event
async def renegotiation_answer(sid, data):
    await socket_handler.renegotiation_answer(sid, data)
    

@sio.event
async def incoming_answer(sid ,data):
    await socket_handler.incoming_answer(sid ,data)

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


 