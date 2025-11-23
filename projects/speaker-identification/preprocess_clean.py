import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Configuration
SAMPLE_RATE = 44100
DURATION = 5
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
DATA_DIR = "data/enhanced"  # Use enhanced audio
PROCESSED_DIR = "data/processed_clean"
IMG_SIZE = (224, 224)

def load_audio(file_path):
    """Load enhanced audio file"""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        target_len = SAMPLE_RATE * DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

# Data Augmentation Functions
def add_noise(y, noise_factor=0.003):
    """Add light noise"""
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def time_shift(y, shift_max=0.1):
    """Shift audio in time"""
    shift = np.random.randint(int(SAMPLE_RATE * shift_max))
    direction = np.random.choice([-1, 1])
    shift = shift * direction
    augmented = np.roll(y, shift)
    if shift > 0:
        augmented[:shift] = 0
    else:
        augmented[shift:] = 0
    return augmented

def change_pitch(y, sr, n_steps=1):
    """Change pitch slightly"""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def change_speed(y, speed_factor=1.05):
    """Change speed slightly"""
    return librosa.effects.time_stretch(y, rate=speed_factor)

def add_background_noise(y, noise_factor=0.002):
    """Add very light background noise"""
    noise = np.random.normal(0, noise_factor, len(y))
    return y + noise

def generate_spectrogram(y, sr):
    """Generate mel-spectrogram"""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def save_spectrogram_image(S_dB, output_path):
    """Save spectrogram as image"""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def augment_and_save(y, sr, base_filename, output_dir):
    """Create augmented versions"""
    augmentations = []
    
    # Original
    augmentations.append((y, "original"))
    
    # Light augmentations (conservative for clean data)
    augmentations.append((add_noise(y, 0.002), "noise_light"))
    augmentations.append((time_shift(y, 0.08), "shift_1"))
    augmentations.append((time_shift(y, 0.12), "shift_2"))
    augmentations.append((change_pitch(y, sr, 0.5), "pitch_up_light"))
    augmentations.append((change_pitch(y, sr, -0.5), "pitch_down_light"))
    augmentations.append((change_speed(y, 0.97), "speed_slow"))
    augmentations.append((change_speed(y, 1.03), "speed_fast"))
    augmentations.append((add_background_noise(y, 0.001), "bg_noise"))
    
    # Combined (light)
    combined1 = add_noise(time_shift(y, 0.08), 0.002)
    augmentations.append((combined1, "combined_1"))
    
    combined2 = add_background_noise(change_speed(y, 0.98), 0.001)
    augmentations.append((combined2, "combined_2"))
    
    # Save all
    saved_count = 0
    for aug_audio, aug_name in augmentations:
        S_dB = generate_spectrogram(aug_audio, sr)
        output_filename = f"{base_filename}_{aug_name}.png"
        output_path = os.path.join(output_dir, output_filename)
        save_spectrogram_image(S_dB, output_path)
        saved_count += 1
    
    return saved_count

def preprocess_clean_dataset():
    """Preprocess enhanced audio dataset"""
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
    
    speakers = sorted([d for d in os.listdir(DATA_DIR) 
                      if os.path.isdir(os.path.join(DATA_DIR, d))])
    
    print("="*60)
    print("CLEAN SPECTROGRAM GENERATION")
    print("="*60)
    print(f"Input: {DATA_DIR} (enhanced audio)")
    print(f"Output: {PROCESSED_DIR}")
    print(f"Found {len(speakers)} speakers: {speakers}")
    print(f"Augmentation: ~11 versions per audio file")
    print("="*60)
    
    total_original = 0
    total_augmented = 0
    
    for speaker in speakers:
        speaker_input_dir = os.path.join(DATA_DIR, speaker)
        speaker_output_dir = os.path.join(PROCESSED_DIR, speaker)
        
        if not os.path.exists(speaker_output_dir):
            os.makedirs(speaker_output_dir)
        
        audio_files = sorted([f for f in os.listdir(speaker_input_dir) if f.endswith(".wav")])
        print(f"\n{speaker}: Processing {len(audio_files)} files...")
        
        for filename in audio_files:
            file_path = os.path.join(speaker_input_dir, filename)
            y, sr = load_audio(file_path)
            
            if y is not None:
                base_filename = filename.replace(".wav", "")
                count = augment_and_save(y, sr, base_filename, speaker_output_dir)
                total_original += 1
                total_augmented += count
                print(f"  ✓ {filename} → {count} spectrograms")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Original audio files: {total_original}")
    print(f"Total spectrograms: {total_augmented}")
    print(f"Augmentation factor: {total_augmented/total_original:.1f}x")
    print(f"Output: {PROCESSED_DIR}/")
    print("="*60)

if __name__ == "__main__":
    preprocess_clean_dataset()
