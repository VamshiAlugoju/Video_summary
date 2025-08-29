import socket
import json
import torch
import torchaudio
from seamless_communication.inference import Translator
import os

# Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = "seamlessM4T_v2_large"
vocoder_name = "vocoder_v2" if model_name == "seamlessM4T_v2_large" else "vocoder_36langs"
translator = Translator(model_name, vocoder_name, device=device, dtype=torch.float16)

host = 'localhost'
port = 51008

# Server socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((host, port))
    s.listen()
    print(f"[Seamless Server] Listening on {host}:{port}")
    print("[Seamless Server] Ready and waiting for translation tasks...")
    while True:
        conn, addr = s.accept()
        with conn:
            print(f"[Seamless Server] Connected by {addr}")

            # Receive and decode JSON
            data = conn.recv(4096).decode()
            try:
                json_data = json.loads(data)
                input_text = json_data["text"]
                tgt_lang = json_data["tgt_lang"]
            except Exception as e:
                print(f"[Seamless Server] Failed to parse JSON: {e}")
                conn.sendall("ERROR: Invalid JSON".encode())
                exit(1)

            #print(f"[Seamless Server] Text: {input_text}")
            #print(f"[Seamless Server] Target language: {tgt_lang}")

            # Translation
            text_output, speech_output = translator.predict(
                input=input_text,
                task_str="t2st",
                tgt_lang=tgt_lang,
                src_lang="eng"
            )

            folder_name = "summary_videos"
            os.makedirs(folder_name, exist_ok=True)

            # Extract audio tensor and sample rate
            audio_tensor = speech_output.audio_wavs[0][0].to(torch.float32).cpu()
            sample_rate = speech_output.sample_rate

            # Ensure shape is [channels, samples]
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # make it [1, samples]

            # Create 1 second of silence with same channel count
            num_silent_samples = sample_rate  # 1 second
            silence = torch.zeros((audio_tensor.shape[0], num_silent_samples), dtype=torch.float32)

            # Concatenate original audio + silence
            audio_with_silence = torch.cat([audio_tensor, silence], dim=-1)

            # Save the new audio
            output_path = os.path.join(folder_name, "translated_audio.wav")
            torchaudio.save(output_path, audio_with_silence, sample_rate)

            conn.sendall(output_path.encode())
            print(f"[Seamless Server] Audio file sent: {output_path}")
