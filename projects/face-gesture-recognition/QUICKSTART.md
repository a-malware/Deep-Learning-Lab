# Quick Start Guide

Get up and running with the Combined Face & Gesture Recognition System in 5 minutes!

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Run the System

```bash
python combined_face_gesture_recognition.py --debug
```

That's it! The system will:
- Open your webcam
- Detect and recognize faces
- Detect hand gestures (peace, thumbs up, stop)
- Associate gestures with people

## Controls

- **'s'** - Toggle debug mode (show/hide bounding boxes)
- **'q' or ESC** - Quit

## Expected Output

```bash
[PERSON] EAC22076 made a THUMBS UP gesture!
[GESTURE] PEACE gesture detected!
[PERSON] EAC22015 made a STOP gesture!
```

## Add Your Face

If the system doesn't recognize you:

```bash
python scripts/retrain_my_face.py
```

Follow the on-screen instructions to capture 120 images of your face.

## Capture Screenshots

To save images for reports or documentation:

```bash
python scripts/capture_report_images.py
```

Press number keys (1-4) to capture different scenarios.

## Troubleshooting

**Camera not opening?**
- Check if another application is using the camera
- Try changing camera index in the code (default is 0)

**No face recognition?**
- Make sure you've trained your face using `retrain_my_face.py`
- Ensure good lighting conditions

**Low FPS?**
- Close other applications
- This is normal for CPU inference (15-30 FPS)

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Generate the IEEE report in `docs/` folder

## Need Help?

Open an issue on GitHub with:
- Your operating system
- Python version
- Error messages
- What you were trying to do

Happy coding!
