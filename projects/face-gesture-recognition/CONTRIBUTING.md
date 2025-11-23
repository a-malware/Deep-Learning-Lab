# Contributing to Combined Face & Gesture Recognition System

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/dynamic_gestures.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes thoroughly
6. Commit with clear messages: `git commit -m "Add feature: description"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Development Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (if any)
pip install black isort flake8
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and concise
- Add comments for complex logic

### Formatting

We use `black` and `isort` for code formatting:

```bash
# Format code
black .
isort .
```

## Testing

Before submitting a PR:

1. Test the main application: `python combined_face_gesture_recognition.py --debug`
2. Test face retraining: `python scripts/retrain_my_face.py`
3. Test image capture: `python scripts/capture_report_images.py`
4. Verify no errors or warnings

## Areas for Contribution

### High Priority
- Add unit tests for core modules
- Improve gesture recognition accuracy
- Add more gesture types
- GPU acceleration support
- Performance optimizations

### Medium Priority
- Better error handling
- Configuration file support
- Logging improvements
- Documentation enhancements
- Additional examples

### Low Priority
- UI improvements
- Additional visualization options
- Export functionality
- Statistics tracking

## Pull Request Guidelines

- **Title**: Clear and descriptive
- **Description**: Explain what changes you made and why
- **Testing**: Describe how you tested your changes
- **Screenshots**: Include if UI changes are involved
- **Documentation**: Update README.md if needed

## Reporting Issues

When reporting issues, please include:

- Operating system and version
- Python version
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages or logs
- Screenshots if applicable

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn and grow

## Questions?

Feel free to open an issue for questions or discussions.

Thank you for contributing!
