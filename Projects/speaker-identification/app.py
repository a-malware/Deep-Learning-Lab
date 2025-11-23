from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from model import SimpleSpeakerCNN
import tempfile
import noisereduce as nr
from scipy import signal

app = Flask(__name__, static_folder='web', static_url_path='')
CORS(app)  # Enable CORS for development

# Configuration
MODEL_PATH = "models/speaker_cnn_final.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 44100
DURATION = 5
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
IMG_SIZE = (224, 224)

# Audio enhancement parameters
TARGET_PEAK_DB = -3.0
TARGET_RMS_DB = -20.0

# Load model at startup
print("Loading model...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
speaker_to_idx = checkpoint['speaker_to_idx']
num_speakers = len(speaker_to_idx)

model = SimpleSpeakerCNN(num_speakers=num_speakers).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Model loaded successfully!")
print(f"  Speakers: {list(speaker_to_idx.keys())}")

# Transform for images
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

def enhance_audio(audio, sr):
    """Apply same enhancement as training data"""
    # Noise reduction
    try:
        audio = nr.reduce_noise(y=audio, sr=sr, stationary=True, 
                               prop_decrease=0.8, freq_mask_smooth_hz=500,
                               time_mask_smooth_ms=50)
    except:
        pass
    
    # Trim silence
    audio_db = librosa.amplitude_to_db(np.abs(audio), ref=np.max)
    non_silent = audio_db > -40
    if np.any(non_silent):
        indices = np.where(non_silent)[0]
        start_idx = max(0, indices[0] - int(0.1 * sr))
        end_idx = min(len(audio), indices[-1] + int(0.1 * sr))
        audio = audio[start_idx:end_idx]
    
    # High-pass filter (80 Hz)
    sos = signal.butter(5, 80, 'hp', fs=sr, output='sos')
    audio = signal.sosfilt(sos, audio)
    
    # De-emphasis filter (8000 Hz)
    sos_deemph = signal.butter(2, 8000, 'lp', fs=sr, output='sos')
    audio = signal.sosfilt(sos_deemph, audio)
    
    # Peak normalization
    peak = np.abs(audio).max()
    if peak > 0:
        target_peak = 10 ** (TARGET_PEAK_DB / 20.0)
        audio = audio * (target_peak / peak)
    
    # RMS normalization
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        target_rms = 10 ** (TARGET_RMS_DB / 20.0)
        audio = audio * (target_rms / rms)
    
    # Clip
    audio = np.clip(audio, -1.0, 1.0)
    
    # Ensure correct duration
    target_length = int(DURATION * sr)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    
    return audio

def process_audio(audio_file):
    """Process audio file and return prediction"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_file, sr=SAMPLE_RATE, duration=DURATION)
        
        # Apply enhancement
        audio = enhance_audio(audio, sr)
        
        # Generate mel-spectrogram
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, 
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Save as temporary image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_img_path = tmp.name
            
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, hop_length=HOP_LENGTH, 
                                x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(temp_img_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Load and transform image
        image = Image.open(temp_img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get results
        idx_to_speaker = {v: k for k, v in speaker_to_idx.items()}
        predicted_speaker = idx_to_speaker[predicted.item()]
        confidence_score = confidence.item() * 100
        
        all_probs = {}
        probs_array = probabilities[0].cpu().numpy()
        for idx, speaker in idx_to_speaker.items():
            all_probs[speaker] = float(probs_array[idx] * 100)
        
        # Cleanup
        os.remove(temp_img_path)
        
        return {
            'success': True,
            'predicted_speaker': predicted_speaker,
            'confidence': confidence_score,
            'all_probabilities': all_probs
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('web', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_audio_path = tmp.name
            audio_file.save(temp_audio_path)
        
        # Process
        result = process_audio(temp_audio_path)
        
        # Cleanup
        os.remove(temp_audio_path)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎙️  SPEAKER IDENTIFICATION WEB INTERFACE")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Speakers: {list(speaker_to_idx.keys())}")
    print(f"Device: {DEVICE}")
    print("\nStarting server...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
