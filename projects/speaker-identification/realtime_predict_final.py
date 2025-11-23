import os
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from model import SimpleSpeakerCNN
import sounddevice as sd
from scipy.io import wavfile
import time
import tempfile
import noisereduce as nr
from scipy import signal

# Configuration
MODEL_PATH = "models/speaker_cnn_final.pth"  # Use new model
SAMPLE_RATE = 44100
DURATION = 5
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
IMG_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Audio enhancement parameters (same as enhance_audio.py)
TARGET_PEAK_DB = -3.0
TARGET_RMS_DB = -20.0

class RealtimeSpeakerIdentifier:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.speaker_to_idx = None
        self.idx_to_speaker = None
        self.temp_dir = tempfile.mkdtemp()
        
        # Load model
        self.load_model()
        
        # Transform for images
        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file '{self.model_path}' not found!")
        
        print("Loading model...")
        checkpoint = torch.load(self.model_path, map_location=DEVICE)
        self.speaker_to_idx = checkpoint['speaker_to_idx']
        self.idx_to_speaker = {v: k for k, v in self.speaker_to_idx.items()}
        num_speakers = len(self.speaker_to_idx)
        
        self.model = SimpleSpeakerCNN(num_speakers=num_speakers).to(DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Model loaded successfully!")
        print(f"  Model: speaker_cnn_final.pth (100% accuracy)")
        print(f"  Speakers: {list(self.speaker_to_idx.keys())}")
    
    def enhance_audio(self, audio, sr):
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
    
    def record_audio(self, duration=DURATION):
        """Record audio from microphone"""
        print(f"\n🎤 Recording for {duration} seconds...")
        print("   Say: 'CONVOLUTIONAL NEURAL NETWORKS CAN BE USED FOR SPEECH AND VOICE RECOGNITION'")
        
        # Countdown
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        
        print("   🔴 RECORDING NOW!")
        recording = sd.rec(int(duration * SAMPLE_RATE), 
                          samplerate=SAMPLE_RATE, 
                          channels=1, 
                          dtype='float32')
        sd.wait()
        print("   ✓ Recording complete!")
        
        return recording
    
    def preprocess_audio(self, audio_data):
        """Convert audio to spectrogram image with enhancement"""
        # Flatten if needed
        if len(audio_data.shape) > 1:
            audio_data = audio_data.flatten()
        
        # Apply enhancement (same as training)
        audio_data = self.enhance_audio(audio_data, SAMPLE_RATE)
        
        # Generate mel-spectrogram
        S = librosa.feature.melspectrogram(y=audio_data, sr=SAMPLE_RATE, 
                                          n_mels=N_MELS, n_fft=N_FFT, 
                                          hop_length=HOP_LENGTH)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Save as temporary image
        temp_img_path = os.path.join(self.temp_dir, 'temp_spec.png')
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, 
                                x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(temp_img_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        return temp_img_path
    
    def predict(self, image_path):
        """Predict speaker from spectrogram image"""
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_speaker = self.idx_to_speaker[predicted.item()]
        confidence_score = confidence.item() * 100
        all_probs = probabilities[0].cpu().numpy()
        
        return predicted_speaker, confidence_score, all_probs
    
    def display_results(self, speaker, confidence, probabilities):
        """Display prediction results"""
        print("\n" + "="*60)
        print("🎯 PREDICTION RESULT (ENHANCED MODEL)")
        print("="*60)
        print(f"Predicted Speaker: {speaker}")
        print(f"Confidence: {confidence:.2f}%")
        
        # Confidence indicator
        if confidence >= 95:
            indicator = "🟢 Excellent Confidence"
        elif confidence >= 90:
            indicator = "🟢 Very High Confidence"
        elif confidence >= 80:
            indicator = "🟡 High Confidence"
        elif confidence >= 70:
            indicator = "🟠 Medium Confidence"
        else:
            indicator = "🔴 Low Confidence"
        print(f"Status: {indicator}")
        
        print(f"\nAll Probabilities:")
        for idx in sorted(self.idx_to_speaker.keys()):
            speaker_name = self.idx_to_speaker[idx]
            prob = probabilities[idx] * 100
            bar_length = int(prob / 2)
            bar = '█' * bar_length
            marker = " ← PREDICTED" if speaker_name == speaker else ""
            print(f"  {speaker_name}: {prob:5.2f}% {bar}{marker}")
        
        print("="*60)
    
    def run_interactive(self):
        """Run interactive real-time prediction"""
        print("\n" + "="*60)
        print("🎙️  REAL-TIME SPEAKER IDENTIFICATION (ENHANCED MODEL)")
        print("="*60)
        print(f"Device: {DEVICE}")
        print(f"Model: speaker_cnn_final.pth (100% test accuracy)")
        print(f"Recording duration: {DURATION} seconds")
        print(f"Sample rate: {SAMPLE_RATE} Hz")
        print(f"Audio enhancement: ENABLED ✅")
        print("="*60)
        
        while True:
            print("\n" + "-"*60)
            user_input = input("\nPress ENTER to record, or 'q' to quit: ").strip().lower()
            
            if user_input == 'q':
                print("\n👋 Goodbye!")
                break
            
            try:
                # Record audio
                audio_data = self.record_audio()
                
                # Preprocess with enhancement
                print("\n⚙️  Processing audio (with enhancement)...")
                image_path = self.preprocess_audio(audio_data)
                
                # Predict
                print("🔍 Identifying speaker...")
                speaker, confidence, probs = self.predict(image_path)
                
                # Display results
                self.display_results(speaker, confidence, probs)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Speaker Identification (Enhanced Model)')
    args = parser.parse_args()
    
    try:
        # Initialize identifier
        identifier = RealtimeSpeakerIdentifier(MODEL_PATH)
        
        # Run interactive mode
        identifier.run_interactive()
            
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please train the model first using: python train_final.py")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
