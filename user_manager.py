import os
import asyncio
from typing import List, Optional, TypedDict 
from aiortc import RTCPeerConnection, RTCRtpSender, RTCSessionDescription , MediaStreamTrack 
import time
from queue import Queue
from VideoSummaryOriginal import video_summary_server
import wave
import math


from audio_player import AudioPlayer, PlayerStreamTrackPlayer
from socket_instance import sio
from constants import lang_map , INTERVAL_SEC , PROCESSING_FPS, TOTAL_INTERVAL
rev_lang_map = {v: k for k, v in lang_map.items()}

ROOT = os.path.dirname(__file__)

def get_wav_length(path: str) -> int:
    with wave.open(path, "rb") as f:
        frames = f.getnframes()
        rate = f.getframerate()
        length = frames / float(rate)
    return math.ceil(length)

class SummaryData(TypedDict):
    summary : str
    audio_file_path : str
    file_duration : float


class User:
    def __init__(self, sid: str ):
        self.sid: str = sid
   
        self.frames= []
        self.is_recording = False
        self.peer_connection_in : RTCPeerConnection = None
        self.peer_connection_out : RTCPeerConnection = None
        self.start_time : float = None
        self.stream_start_time : float = None
        self.stream_end_time : float = None
        self.prev_summary_processed_time : float = None
        self.target_language : str = rev_lang_map["English"]
        self.is_streaming : bool = False
        self.curr_idx : int = 0
        self.summary_queue : Queue[SummaryData] = Queue()
        self.curr_summary : SummaryData = None

        self.video_summary_server = None
        self.is_ready_for_track = True
        self.is_summary_processing = False
        self.curr_process_task = None
        self.read_frames = True
        print("✅ User initialized for socket id" , sid)



    def load_video_summary_server(self):
        """Load video summary server."""
        env= os.getenv("PY_ENV")
        # if env == "dev":
        #     from VideoSummary import video_summary_server
        #     self.video_summary_server = video_summary_server
        # else:
        #     from VideoSummaryOriginal import video_summary_server
        #     self.video_summary_server = video_summary_server

   
    
    def select_target_language(self, language: str):
        """Select target language."""
        self.target_language = language
    
    async  def handle_offer(self, data):
        """Handle WebRTC offer."""
        offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
        pc = RTCPeerConnection()

        @pc.on("connectionstatechange")
        async def on_connection_state_change():
             print(" pc_incoming Connection state changed to:", pc.connectionState, "for socket id" , self.sid)
            


        @pc.on("track")
        async def on_track(track: MediaStreamTrack):
            print("track received", "and track type is " , track.kind)
            await self.handle_incoming_track(track)

            
        
        self.peer_connection_in = pc
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        await sio.emit("answer", {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}, to=self.sid)

    async def initiate_pc_out_connection(self , audio_track_path):
        """Generate WebRTC offer.""" 
        try:
            pc = RTCPeerConnection()
            path = os.path.join(ROOT, audio_track_path)
            player =  AudioPlayer(path , on_ended = self.on_ended , sid = self.sid)
             
            pc.addTrack(player.audio)
            offer = await pc.createOffer()

            await pc.setLocalDescription(offer)
            self.peer_connection_out = pc
            await sio.emit("incoming_offer" , { "sdp": pc.localDescription.sdp, "type": pc.localDescription.type }, to=self.sid)
        except Exception as e:
            print("Error initiating WebRTC connection for socket id" , self.sid , e)
            self.is_streaming = False
            return

    
    async def handle_pc_out_answer(self , data):
        """Handle WebRTC answer."""
        answer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
        if(self.peer_connection_out is None):
            print("peer_connection_out is None")
            return
        await self.peer_connection_out.setRemoteDescription(answer)
         
        @self.peer_connection_out.on("connectionstatechange")
        async def on_connection_state_change():
             print("  pc_out Connection state changed to:", self.peer_connection_out.connectionState, "for socket id" , self.sid)
         
    async def handle_incoming_track(self , track):
        """Handle incoming tracks"""
        print(f"[{self.sid}] Track received: {track.kind}")
        if track.kind == "video":
            async def process_video_track(track):
                """Process incoming video track."""
                stream_start_time = time.time()  # local to this async function
                self.prev_summary_processed_time = time.time()
                self.is_recording = True
                try:
                    while True:
                        
                        frame = await track.recv()
                        self.frames.append(frame)
                        time_diff = time.time() - stream_start_time
                        if time_diff >= INTERVAL_SEC:
                            frames_to_process = self.frames[:] 
                            self.frames = []
                            if self.read_frames is True  and self.is_summary_processing is False:
                              self.curr_process_task =  asyncio.create_task(self.delayed_processing(frames_to_process))
                            else:
                                print(f"skipping current batch of frames is_summary_processing = {self.is_summary_processing}  and is_streaming = {self.is_streaming} ")
                            
                            print("resetting stream start time-----------==++++++++++++++>>" , len(self.frames))
                            stream_start_time = time.time()  # reset timer
                except Exception as e:
                    print(f"Track ended for {self.sid}, stopping frame logging" , e)
                    self.is_recording = False
            asyncio.create_task(process_video_track(track))

        
    # Updated User.delayed_processing method
    async def delayed_processing(self, frames):
        """Process frames after a delay - Updated for multi-user support"""
        if self.is_streaming == True:
            print("programme is still streaming skipping the current frames")
            return
        
        try:
            print(f"⏳ {len(frames)} frames collected for user {self.sid}, processing now...")
            
            # Initialize chunk counter if it doesn't exist
            if not hasattr(self, 'chunk_counter'):
                self.chunk_counter = 0
            if not hasattr(self, 'chunks_for_5sec'):
                self.chunks_for_5sec = TOTAL_INTERVAL  # Number of 1-second chunks to accumulate
            
            # Increment chunk counter
            self.chunk_counter += 1
            
            # Calculate FPS and filter frames as before
            current_fps = len(frames) / INTERVAL_SEC
            self.prev_summary_processed_time = time.time()
            print(f"FPS: {current_fps}")
            
            skip_frames = max(1, int(current_fps / PROCESSING_FPS))
            filtered_frames = []
            for i in range(0, len(frames), skip_frames):
                filtered_frames.append(frames[i])
            
            print(f"🚀 Processing frames size for user {self.sid}:", len(filtered_frames))
            
            # Determine if this is the final chunk for 5-second processing
            is_final_chunk = (self.chunk_counter % self.chunks_for_5sec == 0)
            
            self.is_summary_processing = True
            start_time = time.time()
            
            # Process frames with user-specific 5-second logic
            process_output = await video_summary_server.process_video_frames(
                filtered_frames,
                user_session=self,  # Pass the user session object
                options={"target_language": self.target_language},
                is_final_chunk=is_final_chunk
            )
            
            print(f"Processing time for user {self.sid}: {time.time() - start_time}")
            self.is_summary_processing = False
            
            # Only process summary and audio if we got them (every 5 seconds)
            if process_output.get("summary") and process_output.get("audio_file_path"):
                summary = process_output["summary"]
                audio_file_path = process_output["audio_file_path"]
                duration = get_wav_length(os.path.join(ROOT, audio_file_path))
                self.summary_queue.put({
                    "summary": summary,
                    "audio_file_path": audio_file_path,
                    "file_duration":duration
                })
                
                print(f"✅ Summary added to list for socket id {self.sid} (after {self.chunk_counter} chunks)")
                
                # Start streaming the file as soon as it is created
                if (self.is_recording == True and 
                    self.summary_queue.qsize() == 1 and 
                    self.is_streaming == False):
                    print("---------------------------------------------------playing next song --------------------------------------------------------------")
                    await self.load_next_summary()
            else:
                # Intermediate chunk - just log the buffering status
                total_captions = process_output.get("total_captions", 0)
                print(f"📝 Chunk {self.chunk_counter} processed for user {self.sid}, {total_captions} captions buffered, waiting for 5-second completion...")
            
            return process_output
            
        except Exception as e:
            print(f"Track ended for {self.sid}: {e}")
            self.is_summary_processing = False
            # Reset chunk counter on error
            self.chunk_counter = 0
            
            # Cleanup user session on error
            video_summary_server.cleanup_user_session(self.sid)        

        
    async def load_next_summary(self):
         

        if not  self.summary_queue.empty() and self.is_recording == True :
            self.is_streaming = True
            summary_data = self.summary_queue.get() 
            await self.initiate_pc_out_connection(summary_data["audio_file_path"])

            self.curr_summary = summary_data
            print("sending summary to the client")
            await sio.emit("summary_generated" , {  "summary" : summary_data.get("summary") }, to=self.sid)
 
        else:
            print("✅ All summaries processed for socket id" , self.sid)
            self.is_streaming = False
            await self.close_pc_out_connection()
        
        if self.is_recording == False:
            self.is_streaming = False
            self.summary_queue = Queue()
    

    async def on_ended(self):
        """Handle WebRTC track ended."""
        print("✅ Track ended for socket id" , self.sid)
         
        await self.close_pc_out_connection()
        asyncio.create_task(self.remove_completed_file(self.curr_summary["audio_file_path"]))
        print("peer_connection_out closed for socket id" , self.sid)
        await sio.emit("summary_reading_ended" , { "status" : True  } , to=self.sid )
        await self.load_next_summary()
    
    async def remove_completed_file(self, file_path: str):
        """Remove completed file safely."""
        if not file_path:
            return

        try:
            os.remove(file_path)
            print(f"🗑️ Removed file: {file_path}")
        except FileNotFoundError:
            print(f"⚠️ File already missing: {file_path}")
        except Exception as e:
            print(f"❌ Error removing {file_path}: {e}")
        
    async def close_pc_out_connection(self):
        """Close WebRTC connection."""
        if(self.peer_connection_out is None):
            return
        await self.peer_connection_out.close()
        self.peer_connection_out = None
    
    async def close_pc_in_connection(self):
        """Close WebRTC connection."""
        if(self.peer_connection_in is None):
            return
        await self.peer_connection_in.close()
        self.peer_connection_in = None
    
    async def on_stream_close(self):
        """Handle WebRTC disconnect."""
        await self.close_pc_in_connection()
        await self.close_pc_out_connection()
        self.summary_queue = Queue()
        self.is_streaming = False
        self.frames = []
        self.is_recording = False
        if self.curr_process_task is not None:
            try: 
                self.curr_process_task.cancel()
                print("summarization process stoped after stream is stoppedd")
            except Exception as e:
                print("error closing the task" , e)
        video_summary_server.cleanup_user_session(self.sid)

    

    async def handle_disconnect(self):
        """Handle WebRTC disconnect."""
        await self.close_pc_in_connection()
        await self.close_pc_out_connection()
        self.peer_connection_in = None
        self.peer_connection_out = None
        video_summary_server.cleanup_user_session(self.sid)
        


class UserManagerHandler:
    def __init__(self):
        self.users_by_sid: dict[str, User] = {}

    def add_user(self, sid: str ) -> User:
        """Create a new User and map it by SID."""
        user = User(sid)
        self.users_by_sid[sid] = user
        return user

    def get_user(self, sid: str) -> Optional[User]:
        """Retrieve a User by SID."""
        return self.users_by_sid.get(sid)

    async def remove_user(self, sid: str):
        """Remove a User when they disconnect."""
        if sid in self.users_by_sid:
            await self.users_by_sid[sid].handle_disconnect()
            del self.users_by_sid[sid]

