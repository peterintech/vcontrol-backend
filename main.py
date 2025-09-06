from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from pydub import AudioSegment
import traceback
from typing import List
import json
import os
import requests
import whisper
import re
import numpy as np
from pydantic import BaseModel

# AI-powered imports
from sentence_transformers import SentenceTransformer
import sklearn.metrics.pairwise as pairwise

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

class CommandRequest(BaseModel):
    command: str

# ---------------------------
# Load Models
# ---------------------------
print("Loading models...")
whisper_model = whisper.load_model("base")

# Load Sentence Transformer for semantic similarity
try:
    # Use a lightweight but effective model for semantic similarity
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Semantic similarity model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading semantic model: {e}")
    semantic_model = None

try:
    speaker_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb"
    )
    print("✅ Speaker models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading speaker models: {e}")
    traceback.print_exc()

# ---------------------------
# AI-Powered Voice Command Matcher
# ---------------------------
class AIVoiceCommandMatcher:
    def __init__(self, commands, semantic_model):
        self.commands = commands
        self.semantic_model = semantic_model
        self.command_embeddings = None
        self.command_descriptions = []
        
        if self.semantic_model:
            self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Pre-compute embeddings for all command variations"""
        print("Initializing command embeddings...")
        
        # Generate comprehensive command descriptions
        for cmd in self.commands:
            variations = self._generate_command_descriptions(cmd)
            self.command_descriptions.extend([
                {
                    'command': cmd,
                    'description': desc,
                    'action': self._extract_action_from_desc(desc)
                } for desc in variations
            ])
        
        # Compute embeddings for all descriptions
        descriptions_text = [item['description'] for item in self.command_descriptions]
        self.command_embeddings = self.semantic_model.encode(descriptions_text)
        print(f"✅ Generated {len(self.command_embeddings)} command embeddings")
    
    def _generate_command_descriptions(self, command):
        """Generate natural language descriptions for each command"""
        device_name = command['name'].lower()
        command_id = command['id']
        device_type = command['type']
        
        descriptions = []
        
        # Base command variations
        base_variations = [
            f"turn on {device_name}",
            f"turn off {device_name}",
            f"switch on {device_name}",
            f"switch off {device_name}",
            f"start {device_name}",
            f"stop {device_name}",
            f"enable {device_name}",
            f"disable {device_name}",
            f"activate {device_name}",
            f"deactivate {device_name}",
        ]
        
        # Add device-specific natural descriptions
        if 'light' in command_id:
            light_num = command_id.split('-')[1] if '-' in command_id else ""
            descriptions.extend([
                f"turn on light {light_num}",
                f"turn off light {light_num}",
                f"switch on lamp {light_num}",
                f"switch off lamp {light_num}",
                f"turn on the light in the {device_name.split()[0] if ' ' in device_name else 'room'}",
                f"turn off the light in the {device_name.split()[0] if ' ' in device_name else 'room'}",
            ])
        elif command_id == 'fan':
            descriptions.extend([
                "turn on fan", "turn off fan",
                "start the fan", "stop the fan",
                "turn on ceiling fan", "turn off ceiling fan",
                "start air circulation", "stop air circulation"
            ])
        elif command_id == 'socket':
            descriptions.extend([
                "turn on socket", "turn off socket",
                "turn on power outlet", "turn off power outlet",
                "enable power socket", "disable power socket",
                "turn on electrical outlet", "turn off electrical outlet"
            ])
        elif command_id == 'all-lights':
            descriptions.extend([
                "turn on all lights", "turn off all lights",
                "switch on every light", "switch off every light", 
                "turn on all the lights", "turn off all the lights",
                "illuminate everything", "turn off all lighting",
                "lights on", "lights off",
                "turn on both lights", "turn off both lights"
            ])
        
        descriptions.extend(base_variations)
        return descriptions
    
    def _extract_action_from_desc(self, description):
        """Extract action (on/off) from description"""
        desc_lower = description.lower()
        off_keywords = ['off', 'stop', 'disable', 'deactivate']
        
        for keyword in off_keywords:
            if keyword in desc_lower:
                return 'off'
        return 'on'
    
    def _preprocess_input(self, input_text):
        """Clean and normalize input text"""
        # Remove punctuation and extra spaces
        processed = re.sub(r'[^\w\s]', ' ', input_text.lower())
        processed = re.sub(r'\s+', ' ', processed).strip()
        return processed
    
    def get_best_match(self, voice_input, threshold=0.5):
        """Find best matching command using semantic similarity"""
        if not self.semantic_model or self.command_embeddings is None:
            return self._fallback_matching(voice_input)
        
        try:
            # Preprocess input
            processed_input = self._preprocess_input(voice_input)
            
            # Get embedding for input
            input_embedding = self.semantic_model.encode([processed_input])
            
            # Calculate similarities with all command descriptions
            similarities = pairwise.cosine_similarity(
                input_embedding, 
                self.command_embeddings
            )[0]
            
            # Find best matches
            best_indices = np.argsort(similarities)[::-1]  # Sort descending
            best_matches = []
            
            for idx in best_indices[:5]:  # Top 5 matches
                similarity_score = similarities[idx]
                if similarity_score >= threshold:
                    match_info = self.command_descriptions[idx]
                    best_matches.append({
                        'command': match_info['command'],
                        'action': match_info['action'],
                        'similarity': float(similarity_score),
                        'matched_description': match_info['description'],
                        'confidence': self._get_confidence_level(similarity_score)
                    })
            
            if not best_matches:
                return {
                    'success': False,
                    'message': 'No matching command found',
                    'input': voice_input,
                    'suggestions': self._get_suggestions()
                }
            
            best_match = best_matches[0]
            esp32_command = f"{best_match['action']} {best_match['command']['id'].replace('-', ' ')}"
            
            return {
                'success': True,
                'command': best_match['command'],
                'action': best_match['action'],
                'confidence': best_match['confidence'],
                'similarity_score': best_match['similarity'],
                'matched_description': best_match['matched_description'],
                'esp32_command': esp32_command,
                'alternatives': best_matches[1:3]  # Next 2 best matches
            }
            
        except Exception as e:
            print(f"Error in semantic matching: {e}")
            return self._fallback_matching(voice_input)
    
    def _get_confidence_level(self, score):
        """Convert similarity score to confidence level"""
        if score >= 0.8:
            return 'very high'
        elif score >= 0.7:
            return 'high'
        elif score >= 0.6:
            return 'medium'
        elif score >= 0.5:
            return 'low'
        else:
            return 'very low'
    
    def _get_suggestions(self):
        """Get example commands for user guidance"""
        examples = [
            "turn on living room light",
            "turn off bedroom light", 
            "start the fan",
            "turn on all lights",
            "switch off socket"
        ]
        return examples
    
    def _fallback_matching(self, voice_input):
        """Fallback to basic string matching if AI model fails"""
        print("Using fallback matching...")
        processed_input = self._preprocess_input(voice_input)
        
        # Simple keyword matching as fallback
        for cmd in self.commands:
            cmd_keywords = cmd['name'].lower().split() + [cmd['type']]
            input_words = processed_input.split()
            
            # Check if any command keywords are in input
            matches = sum(1 for keyword in cmd_keywords if any(keyword in word for word in input_words))
            
            if matches > 0:
                action = 'off' if any(word in processed_input for word in ['off', 'stop', 'disable']) else 'on'
                return {
                    'success': True,
                    'command': cmd,
                    'action': action,
                    'confidence': 'low',
                    'similarity_score': 0.5,
                    'matched_description': f"fallback match for {cmd['name']}",
                    'esp32_command': f"{action} {cmd['id'].replace('-', ' ')}"
                }
        
        return {
            'success': False,
            'message': 'No matching command found',
            'suggestions': self._get_suggestions()
        }

# ---------------------------
# Define Commands
# ---------------------------
COMMANDS = [
    {
        "id": "light-1",
        "name": "Living Room Light",
        "type": "light",
        "command": "on light one",
        "icon": "💡",
    },
    {
        "id": "light-2", 
        "name": "Bedroom Light",
        "type": "light",
        "command": "on light two", 
        "icon": "💡",
    },
    {
        "id": "fan",
        "name": "Ceiling Fan",
        "type": "fan",
        "command": "on fan",
        "icon": "🌀",
    },
    {
        "id": "socket",
        "name": "Power Socket",
        "type": "socket",
        "command": "on socket", 
        "icon": "🔌",
    },
    {
        "id": "all-lights",
        "name": "All Lights",
        "type": "lights",
        "command": "on all lights",
        "icon": "💡✨",
    },
]

# Initialize AI-powered command matcher
command_matcher = AIVoiceCommandMatcher(COMMANDS, semantic_model)

# ---------------------------
# Helper Functions
# ---------------------------
def ensure_wav(input_path):
    """Converts input audio to wav if needed."""
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

def preprocess_audio(file_path, target_sr=16000):
    file_path = ensure_wav(file_path)
    signal, fs = torchaudio.load(file_path)
    if fs != target_sr:
        signal = torchaudio.transforms.Resample(orig_freq=fs, new_freq=target_sr)(signal)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    return signal

def get_embedding(file_path):
    signal = preprocess_audio(file_path)
    embedding = speaker_model.encode_batch(signal)
    embedding = embedding.view(-1)
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
    return embedding

# ---------------------------
# ESP32 Communication
# ---------------------------
ESP32_CONTROL_URL = "http://192.168.214.100/control"
ESP32_STATUS_URL = "http://192.168.214.100/status"

def send_command_to_esp32(command: str):
    """Send command to ESP32"""
    try:
        response = requests.post(ESP32_CONTROL_URL, json={"command": command})
        response.raise_for_status()
        return response.json()
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
# API Endpoints
# ---------------------------

@app.post("/control")
async def control(request: CommandRequest):
    """Process voice command using AI-powered semantic matching"""
    try:
        # Use AI-powered semantic matching
        match_result = command_matcher.get_best_match(request.command)
        
        if not match_result['success']:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": match_result['message'],
                    "suggestions": match_result.get('suggestions', [])
                }
            )
        
        # Log match details
        print(f"Input: '{request.command}'")
        print(f"Matched: {match_result['command']['name']}")
        print(f"Similarity: {match_result.get('similarity_score', 'N/A'):.3f}")
        print(f"Confidence: {match_result['confidence']}")
        print(f"ESP32 Command: {match_result['esp32_command']}")
        
        # Send to ESP32
        esp32_response = send_command_to_esp32(match_result['esp32_command'])
        
        if "error" in esp32_response:
            raise HTTPException(status_code=500, detail="Failed to communicate with ESP32.")
        
        return JSONResponse(content={
            "command": request.command,
            "matched_command": match_result['command']['name'],
            "action": match_result['action'],
            "confidence": match_result['confidence'],
            "similarity_score": match_result.get('similarity_score'),
            "matched_description": match_result.get('matched_description'),
            "esp32_command": match_result['esp32_command'],
            "esp32_response": esp32_response,
            "alternatives": [
                {
                    'name': alt['command']['name'],
                    'similarity': alt.get('similarity', 0)
                } for alt in match_result.get('alternatives', [])
            ]
        })
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error processing command: {e}"
        print(error_msg)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/test_match")
async def test_match(request: CommandRequest):
    """Test AI command matching without sending to ESP32"""
    try:
        match_result = command_matcher.get_best_match(request.command)
        print(f"Test Match Input: '{request.command}' => Result: {match_result}")
        return JSONResponse(content={
            "input": request.command,
            "match_result": match_result
        })
    except Exception as e:
        error_msg = f"Error testing match: {e}"
        print(error_msg)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/check_wifi")
async def check_wifi():
    status = check_wifi_status()
    if status is None:
        raise HTTPException(status_code=500, detail="Failed to communicate with ESP32.")
    return JSONResponse(content=status)

# ---------------------------
# Audio Processing Endpoints
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

        path = ensure_wav(path)
        result = whisper_model.transcribe(path)

        return {"transcription": result["text"]}

    except Exception as e:
        error_msg = f"Error in /transcribe: {e}"
        print(error_msg)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": error_msg})