# Quick Setup Guide

## Push to GitHub

### 1. Create a new repository on GitHub
Go to https://github.com/new and create a new repository (don't initialize with README, .gitignore, or license).

### 2. Add remote and push

```bash
# Add your GitHub repository as remote
git remote add origin https://github.com/yourusername/speaker-identification.git

# Push to GitHub
git push -u origin main
```

## Local Setup for Development

### 1. Clone the repository (if not already local)
```bash
git clone https://github.com/yourusername/speaker-identification.git
cd speaker-identification
```

### 2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare your data
Create the following directory structure:
```
data/
├── raw/
│   ├── Speaker1/
│   │   ├── Speaker1_1.wav
│   │   ├── Speaker1_2.wav
│   │   └── ...
│   ├── Speaker2/
│   └── ...
```

### 5. Record audio samples
```bash
python record_audio.py
```

### 6. Preprocess audio
```bash
python preprocess_clean.py
```

### 7. Train the model
```bash
python train_final.py
```

### 8. Test predictions
```bash
# Test all samples
python predict_final.py --test-all

# Real-time prediction
python realtime_predict_final.py

# Web interface
python app.py
```

## Git Configuration (if needed)

If you haven't configured git globally, set your identity:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Notes

- The `.gitignore` file excludes large data files and model weights
- You may want to use Git LFS for model files if sharing trained models
- Update the GitHub URL in this guide with your actual repository URL
