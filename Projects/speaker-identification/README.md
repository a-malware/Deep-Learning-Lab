# Speaker Identification System using CNN

A deep learning-based speaker identification system that uses Convolutional Neural Networks (CNN) to identify speakers from audio recordings. The system converts audio into mel-spectrograms and uses image classification techniques to achieve speaker recognition.

## Features

- **High Accuracy**: 100% test accuracy on trained speakers
- **Real-time Prediction**: Identify speakers from live microphone input
- **Web Interface**: User-friendly Flask-based web application
- **Audio Enhancement**: Noise reduction and audio preprocessing
- **Data Augmentation**: Robust training with augmented audio samples
- **Lightweight Model**: Only 33K parameters for efficient inference

## Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 100% |
| Model Size | 33K parameters |
| Average Confidence | 93.9% |
| Inference Time | ~2-3 seconds |

## Architecture

The system uses a custom CNN architecture (`SimpleSpeakerCNN`) with:
- 3 Convolutional layers with batch normalization
- Global average pooling
- Fully connected layers with dropout
- Softmax output for speaker probabilities

See [CNN_ARCHITECTURE.md](CNN_ARCHITECTURE.md) for detailed architecture information.

## Project Structure

```
.
├── model.py                    # CNN architecture definitions
├── train_final.py              # Training script
├── predict_final.py            # Batch prediction script
├── realtime_predict_final.py   # Real-time microphone prediction
├── app.py                      # Flask web application
├── enhance_audio.py            # Audio enhancement utilities
├── preprocess_clean.py         # Audio preprocessing
├── record_audio.py             # Audio recording utility
├── web/                        # Web interface files
│   ├── index.html
│   ├── app.js
│   └── style.css
├── models/                     # Trained model files
│   └── speaker_cnn_final.pth
└── data/                       # Audio data (not included)
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA (optional, for GPU acceleration)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/speaker-identification.git
cd speaker-identification
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Record Audio Samples

```bash
python record_audio.py
```

Follow the prompts to record audio samples for each speaker.

#### 2. Preprocess Audio

```bash
python preprocess_clean.py
```

This will enhance and prepare audio files for training.

#### 3. Train the Model

```bash
python train_final.py
```

The trained model will be saved to `models/speaker_cnn_final.pth`.

#### 4. Test Predictions

**Batch prediction:**
```bash
python predict_final.py --test-all
```

**Real-time prediction:**
```bash
python realtime_predict_final.py
```

**Web interface:**
```bash
python app.py
```
Then open http://localhost:5000 in your browser.

## How It Works

1. **Audio Capture**: Record 5-second audio samples
2. **Preprocessing**: Apply noise reduction, normalization, and filtering
3. **Feature Extraction**: Convert audio to mel-spectrogram images (224×224)
4. **CNN Classification**: Feed spectrograms through CNN for speaker identification
5. **Prediction**: Output speaker identity with confidence scores

## Results

See [FINAL_RESULTS.md](FINAL_RESULTS.md) for detailed performance metrics and analysis.

Key achievements:
- ✅ 100% accuracy on test set
- ✅ Eliminated overfitting through data augmentation
- ✅ 800x smaller model compared to initial version
- ✅ High confidence predictions (avg 93.9%)

## Technologies Used

- **PyTorch**: Deep learning framework
- **Librosa**: Audio processing and feature extraction
- **Flask**: Web application framework
- **NumPy/SciPy**: Numerical computing
- **Matplotlib**: Visualization
- **scikit-learn**: Machine learning utilities

## Documentation

- [CNN_ARCHITECTURE.md](CNN_ARCHITECTURE.md) - Detailed CNN architecture explanation
- [FINAL_RESULTS.md](FINAL_RESULTS.md) - Performance metrics and results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built as part of a machine learning course project
- Uses mel-spectrogram visualization techniques for audio analysis
- Inspired by image classification approaches applied to audio

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: The trained model and audio data are not included in this repository. You'll need to record your own audio samples and train the model following the instructions above.
