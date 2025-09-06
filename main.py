from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from pydub import AudioSegment
import traceback
from typing import List, Dict
import json
import os
import requests
import whisper
import re
import numpy as np
from pydantic import BaseModel
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

class ESP32Command(BaseModel):
    action: str  # "on" or "off"
    device: str  # device_id

class DeviceState(BaseModel):
    """Model for tracking device states"""
    light_1: bool = False
    light_2: bool = False
    fan: bool = False
    socket: bool = False
    all_lights: bool = False

# ---------------------------
# Load Models
# ---------------------------
print("Loading models...")
whisper_model = whisper.load_model("base")

# Load Sentence Transformer for semantic similarity
try:
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
# Device State Management
# ---------------------------
# In-memory storage for device states (in production, use a database)
device_states = DeviceState()

def update_device_state(device_id: str, action: str):
    """Update the in-memory device state"""
    state = action == "on"
    
    if device_id == "light_1":
        device_states.light_1 = state
    elif device_id == "light_2":
        device_states.light_2 = state
    elif device_id == "fan":
        device_states.fan = state
    elif device_id == "socket":
        device_states.socket = state
    elif device_id == "all_lights":
        device_states.light_1 = state
        device_states.light_2 = state
        device_states.all_lights = state
    
    # Update all_lights state based on individual lights
    if device_id in ["light_1", "light_2"]:
        device_states.all_lights = device_states.light_1 and device_states.light_2

# ---------------------------
# AI-Powered Voice Command Matcher
# ---------------------------
class AIVoiceCommandMatcher:
    def __init__(self, devices, semantic_model):
        self.devices = devices
        self.semantic_model = semantic_model
        self.command_embeddings = None
        self.command_descriptions = []
        
        if self.semantic_model:
            self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Pre-compute embeddings for all command variations"""
        print("Initializing command embeddings...")
        
        # Generate comprehensive command descriptions
        for device in self.devices:
            variations = self._generate_command_descriptions(device)
            self.command_descriptions.extend(variations)
        
        # Compute embeddings for all descriptions
        descriptions_text = [item['description'] for item in self.command_descriptions]
        self.command_embeddings = self.semantic_model.encode(descriptions_text)
        print(f"✅ Generated {len(self.command_embeddings)} command embeddings")
    
    def _generate_command_descriptions(self, device):
        """Generate natural language descriptions for each device"""
        device_name = device['name'].lower()
        device_id = device['id']
        
        descriptions = []
        
        # Common action verbs for turning on/off
        on_verbs = ['turn on', 'switch on', 'enable', 'activate', 'start', 'power on', 'put on']
        off_verbs = ['turn off', 'switch off', 'disable', 'deactivate', 'stop', 'power off', 'shut off', 'shut down']
        
        # Generate ON commands
        for verb in on_verbs:
            descriptions.append({
                'device': device,
                'action': 'on',
                'description': f"{verb} {device_name}"
            })
            
            # Add variations without "the"
            descriptions.append({
                'device': device,
                'action': 'on',
                'description': f"{verb} the {device_name}"
            })
        
        # Generate OFF commands
        for verb in off_verbs:
            descriptions.append({
                'device': device,
                'action': 'off',
                'description': f"{verb} {device_name}"
            })
            
            descriptions.append({
                'device': device,
                'action': 'off',
                'description': f"{verb} the {device_name}"
            })
        
        # Device-specific variations
        if device_id == 'light_1':
            specific_variations = [
                ('on', 'turn on living room light'),
                ('on', 'living room light on'),
                ('on', 'lights on in living room'),
                ('on', 'illuminate living room'),
                ('on', 'brighten living room'),
                ('off', 'turn off living room light'),
                ('off', 'living room light off'),
                ('off', 'lights off in living room'),
                ('off', 'darken living room'),
                ('on', 'turn on light one'),
                ('off', 'turn off light one'),
                ('on', 'turn on first light'),
                ('off', 'turn off first light')
            ]
            for action, desc in specific_variations:
                descriptions.append({
                    'device': device,
                    'action': action,
                    'description': desc
                })
                
        elif device_id == 'light_2':
            specific_variations = [
                ('on', 'turn on bedroom light'),
                ('on', 'bedroom light on'),
                ('on', 'lights on in bedroom'),
                ('on', 'illuminate bedroom'),
                ('on', 'brighten bedroom'),
                ('off', 'turn off bedroom light'),
                ('off', 'bedroom light off'),
                ('off', 'lights off in bedroom'),
                ('off', 'darken bedroom'),
                ('on', 'turn on light two'),
                ('off', 'turn off light two'),
                ('on', 'turn on second light'),
                ('off', 'turn off second light')
            ]
            for action, desc in specific_variations:
                descriptions.append({
                    'device': device,
                    'action': action,
                    'description': desc
                })
                
        elif device_id == 'fan':
            specific_variations = [
                ('on', 'start the fan'),
                ('on', 'fan on'),
                ('on', 'start air circulation'),
                ('on', 'cool the room'),
                ('on', 'start cooling'),
                ('off', 'stop the fan'),
                ('off', 'fan off'),
                ('off', 'stop air circulation'),
                ('off', 'stop cooling')
            ]
            for action, desc in specific_variations:
                descriptions.append({
                    'device': device,
                    'action': action,
                    'description': desc
                })
                
        elif device_id == 'socket':
            specific_variations = [
                ('on', 'enable power outlet'),
                ('on', 'power outlet on'),
                ('on', 'enable socket'),
                ('on', 'socket on'),
                ('on', 'plug on'),
                ('on', 'put on'),
                ('off', 'disable power outlet'),
                ('off', 'power outlet off'),
                ('off', 'disable socket'),
                ('off', 'socket off'),
                ('off', 'plug off'),
                ('off', 'put off'),
            ]
            for action, desc in specific_variations:
                descriptions.append({
                    'device': device,
                    'action': action,
                    'description': desc
                })
                
        elif device_id == 'all_lights':
            specific_variations = [
                ('on', 'turn on all lights'),
                ('on', 'all lights on'),
                ('on', 'turn on every light'),
                ('on', 'switch on all lights'),
                ('on', 'illuminate everything'),
                ('on', 'lights on everywhere'),
                ('on', 'turn on both lights'),
                ('on', 'both lights on'),
                ('off', 'turn off all lights'),
                ('off', 'all lights off'),
                ('off', 'turn off every light'),
                ('off', 'switch off all lights'),
                ('off', 'lights off everywhere'),
                ('off', 'turn off both lights'),
                ('off', 'both lights off'),
                ('off', 'complete darkness'),
                ('off', 'shut down all lights')
            ]
            for action, desc in specific_variations:
                descriptions.append({
                    'device': device,
                    'action': action,
                    'description': desc
                })
        
        return descriptions
    
    def _preprocess_input(self, input_text):
        """Clean and normalize input text"""
        # Convert to lowercase and remove extra spaces
        processed = input_text.lower().strip()
        
        # Fix common speech recognition errors
        replacements = {
            'like': 'light',
            'lights': 'light',
            'fun': 'fan',
            'van': 'fan',
            'socket': 'socket',
            'sockets': 'socket',
            'to': 'two',
            'too': 'two',
            'won': 'one',
            'bedroom': 'bedroom',
            'living room': 'living room',
            'livingroom': 'living room',
            'bed room': 'bedroom'
        }
        
        for old, new in replacements.items():
            processed = processed.replace(old, new)
        
        # Remove punctuation but keep spaces
        processed = re.sub(r'[^\w\s]', ' ', processed)
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed
    
    def get_best_match(self, voice_input, threshold=0.45):
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
            best_indices = np.argsort(similarities)[::-1]
            best_matches = []
            
            for idx in best_indices[:5]:
                similarity_score = similarities[idx]
                if similarity_score >= threshold:
                    match_info = self.command_descriptions[idx]
                    best_matches.append({
                        'device': match_info['device'],
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
                    'processed_input': processed_input,
                    'suggestions': self._get_suggestions()
                }
            
            best_match = best_matches[0]
            
            return {
                'success': True,
                'device': best_match['device'],
                'action': best_match['action'],
                'confidence': best_match['confidence'],
                'similarity_score': best_match['similarity'],
                'matched_description': best_match['matched_description'],
                'esp32_command': {
                    'action': best_match['action'],
                    'device': best_match['device']['id']
                },
                'alternatives': best_matches[1:3]
            }
            
        except Exception as e:
            print(f"Error in semantic matching: {e}")
            traceback.print_exc()
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
            "switch off socket",
            "all lights off",
            "fan on",
            "bedroom light on"
        ]
        return examples
    
    def _fallback_matching(self, voice_input):
        """Enhanced fallback matching with better keyword detection"""
        print("Using fallback matching...")
        processed_input = self._preprocess_input(voice_input)
        words = processed_input.split()
        
        # Detect action
        action = None
        off_keywords = ['off', 'stop', 'disable', 'deactivate', 'shut']
        on_keywords = ['on', 'start', 'enable', 'activate', 'open']
        
        for word in words:
            if word in off_keywords:
                action = 'off'
                break
            elif word in on_keywords:
                action = 'on'
                break
        
        if not action:
            action = 'on'  # Default to 'on' if unclear
        
        # Detect device
        best_device = None
        best_score = 0
        
        for device in self.devices:
            score = 0
            device_keywords = device['name'].lower().split()
            device_keywords.append(device['type'].lower())
            
            # Check for device ID matches
            if 'one' in words or '1' in words:
                if device['id'] == 'light_1':
                    score += 2
            elif 'two' in words or '2' in words:
                if device['id'] == 'light_2':
                    score += 2
            elif 'all' in words:
                if device['id'] == 'all_lights':
                    score += 3
            
            # Check for keyword matches
            for keyword in device_keywords:
                if keyword in processed_input:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_device = device
        
        if best_device and best_score > 0:
            return {
                'success': True,
                'device': best_device,
                'action': action,
                'confidence': 'low',
                'similarity_score': 0.5,
                'matched_description': f"fallback match for {best_device['name']}",
                'esp32_command': {
                    'action': action,
                    'device': best_device['id']
                }
            }
        
        return {
            'success': False,
            'message': 'No matching command found',
            'input': voice_input,
            'processed_input': processed_input,
            'suggestions': self._get_suggestions()
        }

# ---------------------------
# Define Devices
# ---------------------------
DEVICES = [
    {
        "id": "light_1",
        "name": "Living Room Light",
        "type": "light",
        "icon": "💡",
        "location": "Living Room",
        "status": "offline"  
    },
    {
        "id": "light_2", 
        "name": "Bedroom Light",
        "type": "light",
        "icon": "💡",
        "location": "Bedroom",
        "status": "offline"
    },
    {
        "id": "fan",
        "name": "Ceiling Fan",
        "type": "fan",
        "icon": "🌀",
        "location": "Living Room",
        "status": "offline"
    },
    {
        "id": "socket",
        "name": "Power Socket",
        "type": "socket",
        "icon": "🔌",
        "location": "Living Room",
        "status": "offline"
    },
    {
        "id": "all_lights",
        "name": "All Lights",
        "type": "lights",
        "icon": "💡✨",
        "location": "All Rooms",
        "status": "offline"
    },
]

# Initialize AI-powered command matcher
command_matcher = AIVoiceCommandMatcher(DEVICES, semantic_model)

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
ESP32_DEVICES_URL = "http://192.168.214.100/devices"

def send_command_to_esp32(command: Dict[str, str]):
    """Send command to ESP32 in new format"""
    try:
        response = requests.post(
            ESP32_CONTROL_URL, 
            json=command,
            timeout=5
        )
        response.raise_for_status()
        
        # Update local device state after successful command
        update_device_state(command['device'], command['action'])
        
        return response.json()
    except requests.exceptions.Timeout:
        print("Timeout communicating with ESP32")
        return {"error": "timeout", "status": "failed"}
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with ESP32: {e}")
        return {"error": str(e), "status": "failed"}

def check_wifi_status():
    try:
        response = requests.get(ESP32_STATUS_URL, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with ESP32: {e}")
        return None

def get_esp32_device_states():
    """Fetch device states from ESP32"""
    try:
        response = requests.get(ESP32_DEVICES_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Update local device states with ESP32 data
        if 'devices' in data:
            devices = data['devices']
            device_states.light_1 = devices.get('light_1', False)
            device_states.light_2 = devices.get('light_2', False)
            device_states.fan = devices.get('fan', False)
            device_states.socket = devices.get('socket', False)
            device_states.all_lights = devices.get('all_lights', False)
        
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching device states from ESP32: {e}")
        return None

# ---------------------------
# API Endpoints
# ---------------------------

@app.get("/devices")
async def get_devices():
    """Get current state of all devices with detailed information"""
    try:
        esp32_states = get_esp32_device_states()
        
        devices_info = []
        for device in DEVICES:
            device_info = {
                "id": device["id"],
                "name": device["name"],
                "type": device["type"],
                "icon": device["icon"],
                "location": device["location"],
                "status": "online" if esp32_states else "offline"
            }
            
            # Add current state based on device ID
            if device["id"] == "light_1":
                device_info["state"] = device_states.light_1
            elif device["id"] == "light_2":
                device_info["state"] = device_states.light_2
            elif device["id"] == "fan":
                device_info["state"] = device_states.fan
            elif device["id"] == "socket":
                device_info["state"] = device_states.socket
            elif device["id"] == "all_lights":
                device_info["state"] = device_states.all_lights
            
            devices_info.append(device_info)
        
        # Build response
        response = {
            "success": True,
            "devices": devices_info,
            "esp32_connected": esp32_states is not None,
            "last_sync": esp32_states.get("timestamp") if esp32_states else None,
            "wifi_info": {
                "connected": esp32_states.get("wifi_connected", False) if esp32_states else False,
                "ip": esp32_states.get("ip", "Unknown") if esp32_states else "Unknown",
                "rssi": esp32_states.get("rssi", None) if esp32_states else None
            } if esp32_states else None
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        error_msg = f"Error fetching devices: {e}"
        print(error_msg)
        traceback.print_exc()
        
        # Return device info with last known states even if ESP32 is unreachable
        fallback_devices = []
        for device in DEVICES:
            device_info = {
                "id": device["id"],
                "name": device["name"],
                "type": device["type"],
                "icon": device["icon"],
                "location": device["location"],
                "status": "offline",
                "state": False  # Default to off if can't connect
            }
            
            # Use cached states
            if device["id"] == "light_1":
                device_info["state"] = device_states.light_1
            elif device["id"] == "light_2":
                device_info["state"] = device_states.light_2
            elif device["id"] == "fan":
                device_info["state"] = device_states.fan
            elif device["id"] == "socket":
                device_info["state"] = device_states.socket
            elif device["id"] == "all_lights":
                device_info["state"] = device_states.all_lights
            
            fallback_devices.append(device_info)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "devices": fallback_devices,
                "esp32_connected": False,
                "warning": "Using cached device states - ESP32 unreachable",
                "error": str(e)
            }
        )

@app.post("/control")
async def control(request: CommandRequest):
    """Process voice command using AI-powered semantic matching"""
    try:
        # Use AI-powered semantic matching
        match_result = command_matcher.get_best_match(request.command)
        
        if not match_result['success']:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": match_result['message'],
                    "input": request.command,
                    "processed_input": match_result.get('processed_input', ''),
                    "suggestions": match_result.get('suggestions', [])
                }
            )
        
        # Log match details
        print(f"Input: '{request.command}'")
        print(f"Matched: {match_result['device']['name']}")
        print(f"Action: {match_result['action']}")
        print(f"Similarity: {match_result.get('similarity_score', 'N/A'):.3f}")
        print(f"Confidence: {match_result['confidence']}")
        print(f"ESP32 Command: {match_result['esp32_command']}")
        
        # Send to ESP32
        esp32_response = send_command_to_esp32(match_result['esp32_command'])
        
        if "error" in esp32_response:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Failed to communicate with ESP32",
                    "error": esp32_response["error"]
                }
            )
        
        return JSONResponse(content={
            "success": True,
            "command": request.command,
            "matched_device": match_result['device']['name'],
            "action": match_result['action'],
            "confidence": match_result['confidence'],
            "similarity_score": match_result.get('similarity_score'),
            "matched_description": match_result.get('matched_description'),
            "esp32_command": match_result['esp32_command'],
            "esp32_response": esp32_response,
            "alternatives": [
                {
                    'name': alt['device']['name'],
                    'action': alt['action'],
                    'similarity': alt.get('similarity', 0)
                } for alt in match_result.get('alternatives', [])
            ]
        })
        
    except Exception as e:
        error_msg = f"Error processing command: {e}"
        print(error_msg)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Internal server error",
                "error": str(e)
            }
        )

@app.post("/test_match")
async def test_match(request: CommandRequest):
    """Test AI command matching without sending to ESP32"""
    try:
        match_result = command_matcher.get_best_match(request.command)
        
        # More detailed test output
        print(f"\n{'='*50}")
        print(f"Test Match Input: '{request.command}'")
        print(f"Success: {match_result.get('success', False)}")
        if match_result.get('success'):
            print(f"Device: {match_result['device']['name']}")
            print(f"Action: {match_result['action']}")
            print(f"Confidence: {match_result['confidence']}")
            print(f"Similarity: {match_result.get('similarity_score', 'N/A'):.3f}")
        print(f"{'='*50}\n")
        
        return JSONResponse(content={
            "input": request.command,
            "match_result": match_result
        })
    except Exception as e:
        error_msg = f"Error testing match: {e}"
        print(error_msg)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Error in test matching",
                "error": str(e)
            }
        )

@app.get("/check_wifi")
async def check_wifi():
    status = check_wifi_status()
    if status is None:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Failed to communicate with ESP32"
            }
        )
    return JSONResponse(content=status)

@app.post("/enroll")
async def enroll(files: List[UploadFile] = File(...)):
    try:
        embeddings_list = []
        for file in files:
            path = f"temp_{file.filename}"
            with open(path, "wb") as f:
                f.write(await file.read())
            embeddings_list.append(get_embedding(path))
            os.remove(path)  # Clean up temp file

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
        os.remove(path)  # Clean up temp file

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
        os.remove(path)  # Clean up temp file

        return {"transcription": result["text"]}

    except Exception as e:
        error_msg = f"Error in /transcribe: {e}"
        print(error_msg)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": error_msg})