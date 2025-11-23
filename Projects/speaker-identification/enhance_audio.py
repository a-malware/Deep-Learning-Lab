import os
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.io import wavfile
import noisereduce as nr

# Configuration
INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/enhanced"
SAMPLE_RATE = 44100
TARGET_DURATION = 5  # seconds
TARGET_PEAK_DB = -3.0  # Peak normalization target
TARGET_RMS_DB = -20.0  # RMS normalization target

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def reduce_noise(audio, sr):
    """
    Advanced noise reduction using spectral gating
    """
    try:
        # Use first 0.5 seconds as noise profile
        noise_sample = audio[:int(0.5 * sr)]
        
        # Apply noise reduction
        reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=True,
            prop_decrease=0.8,  # Reduce noise by 80%
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50
        )
        return reduced
    except Exception as e:
        print(f"  Warning: Noise reduction failed ({e}), using original audio")
        return audio

def normalize_audio(audio):
    """
    Normalize audio using both peak and RMS normalization
    """
    # Peak normalization
    peak = np.abs(audio).max()
    if peak > 0:
        target_peak = 10 ** (TARGET_PEAK_DB / 20.0)
        audio = audio * (target_peak / peak)
    
    # RMS normalization for consistent loudness
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        target_rms = 10 ** (TARGET_RMS_DB / 20.0)
        audio = audio * (target_rms / rms)
    
    # Ensure no clipping
    audio = np.clip(audio, -1.0, 1.0)
    
    return audio

def trim_silence(audio, sr, threshold_db=-40):
    """
    Remove leading and trailing silence
    """
    # Convert to dB
    audio_db = librosa.amplitude_to_db(np.abs(audio), ref=np.max)
    
    # Find non-silent regions
    non_silent = audio_db > threshold_db
    
    if np.any(non_silent):
        # Find first and last non-silent samples
        indices = np.where(non_silent)[0]
        start_idx = max(0, indices[0] - int(0.1 * sr))  # Keep 0.1s before speech
        end_idx = min(len(audio), indices[-1] + int(0.1 * sr))  # Keep 0.1s after speech
        
        audio = audio[start_idx:end_idx]
    
    return audio

def apply_filters(audio, sr):
    """
    Apply high-pass filter to remove low-frequency noise
    """
    # High-pass filter at 80 Hz (removes rumble, hum)
    sos = signal.butter(5, 80, 'hp', fs=sr, output='sos')
    filtered = signal.sosfilt(sos, audio)
    
    # De-emphasis filter (reduces high-frequency noise)
    sos_deemph = signal.butter(2, 8000, 'lp', fs=sr, output='sos')
    filtered = signal.sosfilt(sos_deemph, filtered)
    
    return filtered

def pad_or_trim(audio, sr, target_duration):
    """
    Ensure audio is exactly target_duration seconds
    """
    target_length = int(target_duration * sr)
    current_length = len(audio)
    
    if current_length < target_length:
        # Pad with silence
        padding = target_length - current_length
        audio = np.pad(audio, (0, padding), mode='constant')
    elif current_length > target_length:
        # Trim to target length
        audio = audio[:target_length]
    
    return audio

def enhance_audio_file(input_path, output_path):
    """
    Complete audio enhancement pipeline
    """
    try:
        # Load audio
        audio, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
        
        # Step 1: Noise reduction
        audio = reduce_noise(audio, sr)
        
        # Step 2: Trim silence
        audio = trim_silence(audio, sr)
        
        # Step 3: Apply filters
        audio = apply_filters(audio, sr)
        
        # Step 4: Normalize
        audio = normalize_audio(audio)
        
        # Step 5: Ensure correct duration
        audio = pad_or_trim(audio, sr, TARGET_DURATION)
        
        # Save enhanced audio
        sf.write(output_path, audio, sr, subtype='PCM_16')
        
        return True
        
    except Exception as e:
        print(f"  Error processing {input_path}: {e}")
        return False

def enhance_dataset():
    """
    Enhance all audio files in the dataset
    """
    print("="*60)
    print("AUDIO ENHANCEMENT PIPELINE")
    print("="*60)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Target duration: {TARGET_DURATION} seconds")
    print("="*60)
    
    # Create output directory
    ensure_dir(OUTPUT_DIR)
    
    # Get all speakers
    speakers = sorted([d for d in os.listdir(INPUT_DIR) 
                      if os.path.isdir(os.path.join(INPUT_DIR, d))])
    
    print(f"\nFound {len(speakers)} speakers: {speakers}")
    
    total_files = 0
    processed_files = 0
    failed_files = 0
    
    for speaker in speakers:
        speaker_input_dir = os.path.join(INPUT_DIR, speaker)
        speaker_output_dir = os.path.join(OUTPUT_DIR, speaker)
        
        ensure_dir(speaker_output_dir)
        
        # Get all WAV files
        audio_files = sorted([f for f in os.listdir(speaker_input_dir) 
                            if f.endswith('.wav')])
        
        print(f"\n{speaker}: Processing {len(audio_files)} files...")
        
        for audio_file in audio_files:
            input_path = os.path.join(speaker_input_dir, audio_file)
            output_path = os.path.join(speaker_output_dir, audio_file)
            
            total_files += 1
            
            if enhance_audio_file(input_path, output_path):
                processed_files += 1
                print(f"  ✓ {audio_file}")
            else:
                failed_files += 1
                print(f"  ✗ {audio_file} FAILED")
    
    print("\n" + "="*60)
    print("ENHANCEMENT COMPLETE")
    print("="*60)
    print(f"Total files: {total_files}")
    print(f"Successfully processed: {processed_files}")
    print(f"Failed: {failed_files}")
    print(f"Success rate: {(processed_files/total_files)*100:.1f}%")
    print(f"\nEnhanced audio saved to: {OUTPUT_DIR}/")
    print("="*60)
    
    return processed_files, failed_files

if __name__ == "__main__":
    try:
        import noisereduce
    except ImportError:
        print("ERROR: noisereduce library not found!")
        print("Installing noisereduce...")
        import subprocess
        subprocess.check_call(["pip", "install", "noisereduce"])
        print("Please run the script again.")
        exit(1)
    
    enhance_dataset()
