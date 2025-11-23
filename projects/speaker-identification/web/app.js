// Global variables
let mediaRecorder;
let audioChunks = [];
let audioContext;
let analyser;
let dataArray;
let animationId;
let isRecording = false;

// DOM Elements
const recordBtn = document.getElementById('recordBtn');
const countdown = document.getElementById('countdown');
const recordingIndicator = document.getElementById('recordingIndicator');
const waveformContainer = document.getElementById('waveformContainer');
const waveformCanvas = document.getElementById('waveformCanvas');
const statusMessage = document.getElementById('statusMessage');
const resultsSection = document.getElementById('resultsSection');
const loadingOverlay = document.getElementById('loadingOverlay');
const speakerName = document.getElementById('speakerName');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceBadge = document.getElementById('confidenceBadge');
const probabilitiesList = document.getElementById('probabilitiesList');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    recordBtn.addEventListener('click', handleRecordClick);
    setupCanvas();
});

// Handle record button click
async function handleRecordClick() {
    if (isRecording) return;

    try {
        await startRecordingSequence();
    } catch (error) {
        showStatus('Error accessing microphone: ' + error.message, 'error');
    }
}

// Start recording sequence with countdown
async function startRecordingSequence() {
    // Hide previous results
    resultsSection.classList.add('hidden');
    statusMessage.textContent = '';
    statusMessage.className = 'status-message';

    // Request microphone access
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    // Show countdown
    recordBtn.classList.add('hidden');
    countdown.classList.remove('hidden');

    // Countdown: 3, 2, 1
    for (let i = 3; i > 0; i--) {
        countdown.querySelector('.countdown-number').textContent = i;
        countdown.querySelector('.countdown-number').style.animation = 'none';
        setTimeout(() => {
            countdown.querySelector('.countdown-number').style.animation = 'scaleIn 0.5s ease-out';
        }, 10);
        await sleep(1000);
    }

    // Start recording
    countdown.classList.add('hidden');
    startRecording(stream);
}

// Start actual recording
function startRecording(stream) {
    isRecording = true;
    audioChunks = [];

    // Show recording indicator
    recordingIndicator.classList.remove('hidden');
    waveformContainer.classList.remove('hidden');

    // Setup MediaRecorder
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
    };

    mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        sendAudioForPrediction(audioBlob);
        stream.getTracks().forEach(track => track.stop());
    };

    mediaRecorder.start();

    // Setup waveform visualization
    setupWaveform(stream);

    // Stop after 5 seconds
    setTimeout(() => {
        stopRecording();
    }, 5000);
}

// Stop recording
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }

    isRecording = false;
    recordingIndicator.classList.add('hidden');

    if (animationId) {
        cancelAnimationFrame(animationId);
    }

    showStatus('Processing audio...', 'info');
}

// Setup waveform visualization
function setupWaveform(stream) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);

    source.connect(analyser);
    analyser.fftSize = 256;

    const bufferLength = analyser.frequencyBinCount;
    dataArray = new Uint8Array(bufferLength);

    drawWaveform();
}

// Draw waveform animation
function drawWaveform() {
    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');

    // Set canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    const WIDTH = canvas.width;
    const HEIGHT = canvas.height;

    animationId = requestAnimationFrame(drawWaveform);

    analyser.getByteFrequencyData(dataArray);

    // Clear canvas
    ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
    ctx.fillRect(0, 0, WIDTH, HEIGHT);

    // Draw bars
    const barWidth = (WIDTH / dataArray.length) * 2.5;
    let barHeight;
    let x = 0;

    for (let i = 0; i < dataArray.length; i++) {
        barHeight = (dataArray[i] / 255) * HEIGHT * 0.8;

        // Gradient color based on height
        const gradient = ctx.createLinearGradient(0, HEIGHT - barHeight, 0, HEIGHT);
        gradient.addColorStop(0, '#6366f1');
        gradient.addColorStop(0.5, '#8b5cf6');
        gradient.addColorStop(1, '#ec4899');

        ctx.fillStyle = gradient;
        ctx.fillRect(x, HEIGHT - barHeight, barWidth, barHeight);

        x += barWidth + 1;
    }
}

// Setup canvas
function setupCanvas() {
    const canvas = waveformCanvas;
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
}

// Send audio to backend for prediction
async function sendAudioForPrediction(audioBlob) {
    loadingOverlay.classList.remove('hidden');
    waveformContainer.classList.add('hidden');

    try {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');

        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const result = await response.json();
        displayResults(result);

    } catch (error) {
        showStatus('Error: ' + error.message, 'error');
        console.error('Prediction error:', error);
    } finally {
        loadingOverlay.classList.add('hidden');
        recordBtn.classList.remove('hidden');
    }
}

// Display prediction results
function displayResults(result) {
    if (!result.success) {
        showStatus('Prediction failed: ' + (result.error || 'Unknown error'), 'error');
        return;
    }

    // Show results section
    resultsSection.classList.remove('hidden');

    // Update predicted speaker
    speakerName.textContent = result.predicted_speaker;
    confidenceValue.textContent = result.confidence.toFixed(2) + '%';

    // Update confidence badge color
    if (result.confidence >= 95) {
        confidenceBadge.style.background = 'rgba(16, 185, 129, 0.2)';
        confidenceBadge.style.borderColor = '#10b981';
        confidenceBadge.style.color = '#10b981';
    } else if (result.confidence >= 80) {
        confidenceBadge.style.background = 'rgba(245, 158, 11, 0.2)';
        confidenceBadge.style.borderColor = '#f59e0b';
        confidenceBadge.style.color = '#f59e0b';
    } else {
        confidenceBadge.style.background = 'rgba(239, 68, 68, 0.2)';
        confidenceBadge.style.borderColor = '#ef4444';
        confidenceBadge.style.color = '#ef4444';
    }

    // Display all probabilities
    displayProbabilities(result.all_probabilities, result.predicted_speaker);

    // Show success message
    showStatus(`Identified as ${result.predicted_speaker} with ${result.confidence.toFixed(1)}% confidence`, 'success');

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Display probability bars for all speakers
function displayProbabilities(probabilities, predictedSpeaker) {
    probabilitiesList.innerHTML = '';

    // Sort by probability (highest first)
    const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

    sorted.forEach(([speaker, probability]) => {
        const item = document.createElement('div');
        item.className = 'probability-item';
        if (speaker === predictedSpeaker) {
            item.classList.add('predicted');
        }

        item.innerHTML = `
            <span class="probability-name">${speaker}</span>
            <div class="probability-bar-container">
                <div class="probability-bar" style="width: 0%"></div>
            </div>
            <span class="probability-value">${probability.toFixed(2)}%</span>
        `;

        probabilitiesList.appendChild(item);

        // Animate bar
        setTimeout(() => {
            const bar = item.querySelector('.probability-bar');
            bar.style.width = probability + '%';
        }, 100);
    });
}

// Show status message
function showStatus(message, type = 'info') {
    statusMessage.textContent = message;
    statusMessage.className = `status-message ${type}`;
}

// Utility: sleep function
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Handle window resize
window.addEventListener('resize', () => {
    if (waveformCanvas) {
        setupCanvas();
    }
});
