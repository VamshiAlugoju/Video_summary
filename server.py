
from dotenv import load_dotenv
load_dotenv()


import signal
import sys
import atexit

import asyncio
from aiohttp import web
  
import os
 
import shutil
 
from user_manager import UserManagerHandler  
from socket_instance import sio

user_manager_instance = UserManagerHandler()


ROOT = os.path.dirname(__file__)
 

# ----------------------------
# Socket.IO server setup
# ----------------------------
 
app = web.Application()
sio.attach(app)

 
# ----------------------------
# Socket.IO events
# ----------------------------
@sio.event
async def connect(sid, environ):
    user_manager_instance.add_user(sid)

@sio.event
async def disconnect(sid):
    await user_manager_instance.remove_user(sid)
    

@sio.event
async def stop_stream(sid): 
    user = user_manager_instance.get_user(sid)
    await user.on_stream_close()

@sio.event
async def select_target_language(sid , data):
    user = user_manager_instance.get_user(sid)
    await user.select_target_language(data)
    

@sio.event
async def offer(sid, data):
    user = user_manager_instance.get_user(sid)
    await user.handle_offer(data)

@sio.event
async def incoming_answer(sid ,data):
    user = user_manager_instance.get_user(sid)
    await user.handle_pc_out_answer(data)

# ----------------------------
# App startup
# ----------------------------
async def on_startup(app):    
    print("✅ VideoSummaryServer initialized after server start")
    os.makedirs(os.path.join(ROOT, "audios"), exist_ok=True)
    print(f"PY_ENV is set to {os.getenv('PY_ENV')}")
app.on_startup.append(on_startup)



# --- Your shutdown/cleanup function ---
async def cleanup():
    print("🛑 Cleaning up before exit...")
    audios_path = os.path.join(ROOT, "summary_videos")

    if os.path.exists(audios_path):
        try:
            for filename in os.listdir(audios_path):
                file_path = os.path.join(audios_path, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            print("🗑️ All files inside audios folder deleted.")
        except Exception as e:
            print(f"⚠️ Error cleaning audios folder: {e}")
    else:
        print("ℹ️ No audios folder found, skipping...")

    print("✅ Cleanup finished.")

# --- Graceful exit wrapper ---
def handle_exit(sig=None, frame=None):
    """Called on program exit or signal (Ctrl+C, SIGTERM)."""
    print(f"\n⚡ Received exit signal: {sig}, shutting down...")
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            return  # loop is already gone, do nothing

        if loop.is_running():
            loop.create_task(cleanup())
        else:
            loop.run_until_complete(cleanup())
    except Exception as e:
        print(f"Error during shutdown: {e}")
    finally:
        sys.exit(0)

# --- Register signals ---
signal.signal(signal.SIGINT, handle_exit)   # Ctrl+C
signal.signal(signal.SIGTERM, handle_exit) # kill
 

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    web.run_app(app, port=8080)


 