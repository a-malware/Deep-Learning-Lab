import os
import sys
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from model import SimpleSpeakerCNN
import noisereduce as nr
from scipy import signal

# Configuration
MODEL_PATH = "models/speaker_cnn_final.pth"  # NEW MODEL
SAMPLE_RATE = 44100
DURATION = 5
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
IMG_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Audio enhancement parameters
TARGET_PEAK_DB = -3.0
TARGET_RMS_DB = -20.0

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

def predict_audio(audio_path, model, speaker_to_idx, transform):
    """Predict speaker from audio file"""
    # Load audio
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
    
    # Apply enhancement
    y = enhance_audio(y, sr)
    
    # Generate mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, 
                                      n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Save as temporary image
    import tempfile
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
    
    # Cleanup
    os.remove(temp_img_path)
    
    # Get results
    idx_to_speaker = {v: k for k, v in speaker_to_idx.items()}
    predicted_speaker = idx_to_speaker[predicted.item()]
    confidence_score = confidence.item() * 100
    
    all_probs = {}
    probs_array = probabilities[0].cpu().numpy()
    for idx, speaker in idx_to_speaker.items():
        all_probs[speaker] = float(probs_array[idx] * 100)
    
    return predicted_speaker, confidence_score, all_probs

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_final.py <audio_file.wav>")
        print("   or: python predict_final.py --test-all")
        sys.exit(1)
    
    print("="*60)
    print("SPEAKER IDENTIFICATION - FINAL MODEL")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    speaker_to_idx = checkpoint['speaker_to_idx']
    num_speakers = len(speaker_to_idx)
    
    model = SimpleSpeakerCNN(num_speakers=num_speakers).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✓ Model loaded successfully!")
    print(f"  Speakers: {list(speaker_to_idx.keys())}")
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Test all speakers
    if sys.argv[1] == '--test-all':
        print("\n" + "="*60)
        print("TESTING ALL SPEAKERS")
        print("="*60)
        
        data_dir = "data/raw"
        speakers = sorted([d for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d))])
        
        total = 0
        correct = 0
        
        for speaker in speakers:
            speaker_dir = os.path.join(data_dir, speaker)
            audio_files = sorted([f for f in os.listdir(speaker_dir) if f.endswith('.wav')])
            
            print(f"\n{speaker}: Testing {len(audio_files)} files...")
            
            for audio_file in audio_files:
                audio_path = os.path.join(speaker_dir, audio_file)
                predicted, confidence, _ = predict_audio(audio_path, model, speaker_to_idx, transform)
                
                total += 1
                if predicted == speaker:
                    correct += 1
                    status = "✓"
                else:
                    status = "✗"
                
                print(f"  {status} {audio_file}: {predicted} ({confidence:.2f}%)")
        
        print("\n" + "="*60)
        print("OVERALL RESULTS")
        print("="*60)
        print(f"Total files: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {(correct/total)*100:.2f}%")
        print("="*60)
    
    else:
        # Single file prediction
        audio_path = sys.argv[1]
        
        if not os.path.exists(audio_path):
            print(f"\nError: File '{audio_path}' not found!")
            sys.exit(1)
        
        print(f"\nProcessing: {audio_path}")
        
        predicted, confidence, all_probs = predict_audio(audio_path, model, speaker_to_idx, transform)
        
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"Predicted Speaker: {predicted}")
        print(f"Confidence: {confidence:.2f}%")
        
        print(f"\nAll Probabilities:")
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        for speaker, prob in sorted_probs:
            bar_length = int(prob / 2)
            bar = '█' * bar_length
            marker = " ← PREDICTED" if speaker == predicted else ""
            print(f"  {speaker}: {prob:5.2f}% {bar}{marker}")
        
        print("="*60)

if __name__ == "__main__":
    main()
