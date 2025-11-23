import sounddevice as sd
from scipy.io.wavfile import write
import os
import time

def record_audio(duration, fs, filename):
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write(filename, fs, recording)  # Save as WAV file 
    print(f"Recording saved to {filename}")

def main():
    # Configuration
    fs = 44100  # Sample rate
    duration = 5  # Duration of recording in seconds (adjust as needed for the phrase)
    base_dir = "data/raw"
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    print("Speaker Identification Data Collector")
    print("-------------------------------------")
    print("Phrase to record: 'CONVOLUTIONAL NEURAL NETWORKS CAN BE USED FOR SPEECH AND VOICE RECOGNITION'")
    
    while True:
        speaker_name = input("\nEnter Speaker Name (or 'q' to quit): ").strip()
        if speaker_name.lower() == 'q':
            break
        
        speaker_dir = os.path.join(base_dir, speaker_name)
        if not os.path.exists(speaker_dir):
            os.makedirs(speaker_dir)
        
        # Count existing files to name the new one
        existing_files = [f for f in os.listdir(speaker_dir) if f.endswith('.wav')]
        file_count = len(existing_files)
        filename = os.path.join(speaker_dir, f"{speaker_name}_{file_count + 1}.wav")
        
        input(f"Press Enter to start recording for {speaker_name}...")
        record_audio(duration, fs, filename)
        
        # Option to re-record or continue
        retry = input("Recording complete. Press 'r' to retry, or Enter to continue: ")
        if retry.lower() == 'r':
            os.remove(filename)
            print("Recording discarded.")
            continue

if __name__ == "__main__":
    main()
