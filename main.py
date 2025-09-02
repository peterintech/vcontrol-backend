from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference import EncoderDecoderASR
from pydub import AudioSegment
import traceback
from typing import List
import json
import os
import requests 
from fuzzywuzzy import process  # Fuzzy string matching
from pydantic import BaseModel

# ---------------------------
# App Initialization
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify ["http://localhost:5173"] for stricter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a Pydantic model to receive JSON data
class CommandRequest(BaseModel):
    command: str

# ---------------------------
# Load Models
# ---------------------------
try:
    speaker_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb"
    )
    asr_model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-conformer-transformerlm-librispeech"
    )
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    traceback.print_exc()

# ---------------------------
# Helper: Ensure WAV format
# ---------------------------
def ensure_wav(input_path):
    """
    Converts input audio to wav if needed.
    Returns path to wav file.
    """
    ext = os.path.splitext(input_path)[-1].lower()
    if ext == ".wav":
        return input_path

    wav_path = input_path.rsplit(".", 1)[0] + ".wav"
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        raise RuntimeError(f"Failed to convert {input_path} to wav: {e}")

# ---------------------------
# Audio Preprocessing
# ---------------------------
def preprocess_audio(file_path, target_sr=16000):
    file_path = ensure_wav(file_path)  # 🔑 make sure it’s wav
    signal, fs = torchaudio.load(file_path)
    if fs != target_sr:
        signal = torchaudio.transforms.Resample(orig_freq=fs, new_freq=target_sr)(signal)
    if signal.shape[0] > 1:  # stereo → mono
        signal = signal.mean(dim=0, keepdim=True)
    return signal

# ---------------------------
# Embedding Extraction
# ---------------------------
def get_embedding(file_path):
    signal = preprocess_audio(file_path)
    embedding = speaker_model.encode_batch(signal)
    embedding = embedding.view(-1)
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
    return embedding

# ---------------------------
# Helper Function to Send Command to ESP32
# ---------------------------
ESP32_CONTROL_URL = "http://192.168.214.100/control"
ESP32_STATUS_URL = "http://192.168.214.100/status"  

def send_command_to_esp32(command: str):
    """
    Sends the given command to the ESP32 web server.
    """
    try:
        response = requests.post(ESP32_CONTROL_URL, json={"command": command})
        response.raise_for_status()  # Raise an exception for 4xx/5xx responses
        return response.json()  # Return the response from ESP32
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with ESP32: {e}")
        return {"error": str(e), "status": "failed"} 

def check_wifi_status():
    try:
        response = requests.get(ESP32_STATUS_URL)
        response.raise_for_status()  
        return response.json() 
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with ESP32: {e}")
        return None

# ---------------------------
# Command Processing Endpoint
# ---------------------------
valid_commands = [  
    "on socket", "off socket",
    "on light one", "off light one",
    "on light two", "off light two",
    "on fan", "off fan"]

# Esp32 command control endpoint
@app.post("/control")
async def control(request: CommandRequest):
    # Normalize the command to lowercase and strip extra spaces
    normalized_command = request.command.strip().lower()

    # Use fuzzy matching to find the closest match from the predefined commands
    closest_match, score = process.extractOne(normalized_command, valid_commands)

    # If the score is less than a threshold (e.g., 70), we consider it a bad match
    if score < 80:
        raise HTTPException(status_code=400, detail="Command not recognized. Please speak clearly.")

    # Send the closest matching command to the ESP32
    result = send_command_to_esp32(closest_match)

    if result is None:
        raise HTTPException(status_code=500, detail="Failed to communicate with ESP32.")

    # Return the result from the ESP32 response
    return JSONResponse(content={"command": closest_match, "response": result})

# Endpoint to check Wi-Fi status
@app.get("/check_wifi")
async def check_wifi():
    status = check_wifi_status()
    if status is None:
        raise HTTPException(status_code=500, detail="Failed to communicate with ESP32.")
    
    # Return the Wi-Fi status from the ESP32
    return JSONResponse(content=status)

# ---------------------------
# Endpoints for Audio-related Functions
# ---------------------------

@app.post("/enroll")
async def enroll(files: List[UploadFile] = File(...)):
    try:
        embeddings_list = []
        for file in files:
            path = f"temp_{file.filename}"
            with open(path, "wb") as f:
                f.write(await file.read())
            embeddings_list.append(get_embedding(path))

        average_embedding = torch.stack(embeddings_list).mean(dim=0)
        average_embedding = torch.nn.functional.normalize(average_embedding, p=2, dim=0)

        return {"average_embedding": average_embedding.tolist()}
    except Exception as e:
        error_msg = f"Error in /enroll: {e}"
        print(error_msg)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": error_msg})

@app.post("/verify")
async def verify(
    file: UploadFile = File(...),
    embedding_json: str = Form(...),
):
    try:
        ref_embedding = torch.tensor(json.loads(embedding_json))

        path = f"temp_{file.filename}"
        with open(path, "wb") as f:
            f.write(await file.read())

        new_embedding = get_embedding(path)

        similarity = torch.nn.functional.cosine_similarity(
            ref_embedding.unsqueeze(0),
            new_embedding.unsqueeze(0),
            dim=1
        ).item()

        match = similarity >= 0.4
        return {"similarity": similarity, "match": match}

    except Exception as e:
        error_msg = f"Error in /verify: {e}"
        print(error_msg)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": error_msg})

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        path = f"temp_{file.filename}"
        with open(path, "wb") as f:
            f.write(await file.read())

        path = ensure_wav(path)  # 🔑 make sure it’s wav

        text = asr_model.transcribe_file(path)
        return {"transcription": text}

    except Exception as e:
        error_msg = f"Error in /transcribe: {e}"
        print(error_msg)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": error_msg})
