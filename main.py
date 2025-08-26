from fastapi import FastAPI, File, UploadFile, Form
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

# ---------------------------
# App Initialization
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# Endpoints
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
    embedding_json: str = Form(...)
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

        match = similarity >= 0.5
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
