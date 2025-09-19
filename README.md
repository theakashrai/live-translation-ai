# Live Translation AI

A locally-run, privacy-first live translation tool that provides real-time translation capabilities for audio input, text, and files without requiring internet connectivity or cloud services.

## ✨ Features

- **🔒 Privacy-First**: All processing happens locally on your machine
- **⚡ Real-Time**: Sub-second translation latency for live audio
- **🎯 Multi-Modal**: Support for audio, text, and file inputs
- **🌍 Multi-Language**: Support for 12+ major languages
- **� Voice Cloning**: Clone voices and synthesize translated speech using XTTS
- **�🏗️ Modern Architecture**: Built with Python 3.11+, Poetry, Pydantic, and type safety
- **📱 Easy CLI**: Simple command-line interface with rich output

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Poetry (recommended) or pip
- A microphone for audio translation
- At least 4GB RAM for base models

### Installation

#### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/live-translation-ai.git
cd live-translation-ai

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

#### Using pip

```bash
# Clone the repository
git clone https://github.com/your-username/live-translation-ai.git
cd live-translation-ai

# Install dependencies
pip install -e .
```

### First Run

```bash
# Check system status and available models
translate status

# List available audio devices
translate devices

# Translate text
translate text "Hello, world!" --target es

# Start live audio translation
translate audio --source en --target es --duration 30
```

### Quick English to Hindi Translation Examples

```bash
# Simple text translation (English to Hindi)
translate text "Hello, how are you?" --source en --target hi

# Common phrases
translate text "Good morning!" --source en --target hi
translate text "Thank you very much" --source en --target hi
translate text "Nice to meet you" --source en --target hi

# Interactive mode for multiple translations
translate text --interactive --source en --target hi

# Live audio translation (30 seconds)
translate audio --source en --target hi --duration 30

# Translate from stdin (useful for piping)
echo "Welcome to our application" | translate text --target hi
```

### Voice Cloning Examples

```bash
# Enable voice cloning in settings
export TRANSLATION_VOICE_CLONING_ENABLED=true

# Clone voice and synthesize translated text
translate voice-clone "Hello, how are you today?" \
  --source en --target es \
  --reference examples/my_voice.wav \
  --output cloned_spanish.wav

# Use voice cloning with text command
translate text "Good morning!" \
  --source en --target fr \
  --voice-clone examples/reference_voice.wav \
  --save-audio morning_fr.wav

# Interactive mode with voice cloning
translate text --interactive \
  --source en --target de \
  --voice-clone examples/reference_voice.wav

# Check voice cloning status
translate status  # Shows if TTS is available

# Run voice cloning demo
./examples/voice_cloning_demo.sh
```

## 📖 Usage

### Command Line Interface

The `translate` command provides several subcommands:

#### Text Translation

```bash
# Translate a single text
translate text "Hello, how are you?" --source en --target es

# Interactive mode
translate text --interactive --source en --target fr

# Translate from stdin (useful for piping)
echo "Hello world" | translate text --target es
```

#### Audio Translation

```bash
# Live audio translation (30 seconds)
translate audio --source en --target es --duration 30

# Use specific audio device
translate audio --device 1 --target fr

# Auto-detect source language
translate audio --source auto --target de
```

#### File Translation

```bash
# Translate audio files
translate file audio1.wav audio2.mp3 --target es --output ./translations

# Batch process multiple files
translate file *.wav --source en --target fr
```

#### Configuration

```bash
# Show current configuration
translate config

# Check system status
translate status

# List audio devices
translate devices
```

### Supported Languages

| Code | Language   |
|------|------------|
| en   | English    |
| es   | Spanish    |
| fr   | French     |
| de   | German     |
| it   | Italian    |
| pt   | Portuguese |
| ru   | Russian    |
| zh   | Chinese    |
| ja   | Japanese   |
| ko   | Korean     |
| ar   | Arabic     |
| hi   | Hindi      |
| auto | Auto-detect|

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in your project root:

```env
# Model Settings
TRANSLATION_WHISPER_MODEL=base
TRANSLATION_TRANSLATION_MODEL=nllb-200-distilled-600M
TRANSLATION_DEVICE=cpu

# Audio Settings
TRANSLATION_SAMPLE_RATE=16000
TRANSLATION_CHUNK_LENGTH=30

# Performance Settings
TRANSLATION_BATCH_SIZE=1
TRANSLATION_NUM_THREADS=4

# Language Defaults
TRANSLATION_DEFAULT_SOURCE_LANG=auto
TRANSLATION_DEFAULT_TARGET_LANG=en

# Logging
TRANSLATION_LOG_LEVEL=INFO
TRANSLATION_CACHE_DIR=~/.cache/live-translation-ai
```

### Model Configuration

#### Whisper Models (Speech-to-Text)

| Model  | Size | Speed | Accuracy | Memory |
|--------|------|-------|----------|--------|
| tiny   | 39MB | Fastest | Good     | ~1GB   |
| base   | 74MB | Fast    | Better   | ~1GB   |
| small  | 244MB| Medium  | Good     | ~2GB   |
| medium | 769MB| Slow    | Better   | ~5GB   |
| large  | 1550MB| Slowest| Best     | ~10GB  |

#### Device Configuration

- **CPU**: Works on all systems, slower but reliable
- **CUDA**: Requires NVIDIA GPU with CUDA support, much faster
- **MPS**: Apple Silicon Macs (M1/M2), faster than CPU

## 🏗️ Architecture

### Project Structure

```
live-translation-ai/
├── src/live_translation/
│   ├── core/              # Core models and configuration
│   │   ├── config.py      # Pydantic settings
│   │   ├── models.py      # Data models
│   │   └── exceptions.py  # Custom exceptions
│   ├── translation/       # Translation engines
│   │   ├── engine.py      # Base classes and pipeline
│   │   ├── whisper_document_loader.py  # Whisper LangChain integration
│   │   └── text_translator.py # Text translation
│   ├── audio/             # Audio processing
│   │   ├── capture.py     # Audio capture
│   │   └── processor.py   # Audio processing pipeline
│   ├── api/               # CLI and interfaces
│   │   └── cli.py         # Command-line interface
│   └── utils/             # Utilities
│       └── logger.py      # Structured logging
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
└── docs/                  # Documentation
```

### Data Flow

1. **Input**: Audio/text input via CLI
2. **Processing**:
   - Audio → Whisper → Text transcription
   - Text → Language detection (if auto)
   - Text → Translation model → Translated text
3. **Output**: Formatted translation result

## 🛠️ Development

### Setup Development Environment

```bash
# Install development dependencies
poetry install --with dev

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Type checking
mypy src/

# Code formatting
black src/ tests/
ruff check src/ tests/
```

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Skip slow tests
pytest -m "not slow"

# With coverage report
pytest --cov=src --cov-report=term-missing
```

### Performance Testing

```bash
# Run performance benchmarks
pytest tests/integration/test_translation_workflow.py::TestPerformanceRequirements -v
```

## 📊 Performance

### Latency Requirements (MVP)

- **Text Translation**: < 100ms for texts under 500 characters
- **Audio Translation**: < 500ms from speech end to translation display
- **Memory Usage**: < 4GB RAM with base models
- **CPU Usage**: Optimized for multi-core systems

### Benchmarks

Performance will vary based on:
- Model size (tiny/base/small/medium/large)
- Hardware (CPU/GPU/Apple Silicon)
- Input length and complexity
- Language pair difficulty

## 🔧 Troubleshooting

### Common Issues

#### Installation Problems

```bash
# If you get module import errors
poetry install --no-cache

# For M1/M2 Macs with dependency issues
conda install pytorch torchaudio -c pytorch
poetry install
```

#### Audio Issues

```bash
# List audio devices
translate devices

# Check audio permissions on macOS
# System Preferences → Security & Privacy → Microphone

# Test audio capture
translate audio --device 0 --duration 5
```

#### Model Loading Issues

```bash
# Check model cache
ls -la ~/.cache/live-translation-ai/models/

# Clear cache and reload
rm -rf ~/.cache/live-translation-ai/models/
translate status  # This will trigger model download
```

#### Memory Issues

```bash
# Use smaller models
export TRANSLATION_WHISPER_MODEL=tiny
export TRANSLATION_TRANSLATION_MODEL=nllb-200-distilled-600M

# Monitor memory usage
translate status
```

### Performance Optimization

1. **Use smaller models** for better performance
2. **Enable GPU acceleration** if available (CUDA/MPS)
3. **Increase chunk size** for batch processing
4. **Use SSD storage** for model cache

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest`
6. Submit a pull request

### Code Standards

- **Type hints**: All functions must have type annotations
- **Documentation**: Docstrings for all public functions
- **Testing**: Minimum 80% test coverage
- **Code quality**: Pass ruff, mypy, and black checks

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI Whisper**: Excellent speech-to-text capabilities
- **Hugging Face Transformers**: Translation model infrastructure
- **Facebook NLLB**: Multilingual translation models
- **Rich**: Beautiful terminal output
- **Pydantic**: Data validation and settings management

## 🗺️ Roadmap

### Current (MVP - v0.1.0)
- ✅ Core translation pipeline
- ✅ CLI interface
- ✅ Real-time audio processing
- ✅ Multi-language support

### Next Release (v0.2.0)
- 🔲 GUI Application (PyQt/Tkinter)
- 🔲 Conversation mode (two-way translation)
- 🔲 Custom model fine-tuning
- 🔲 Subtitle generation (SRT export)

### Future (v1.0.0+)
- 🔲 Multiple speaker support
- 🔲 Screen text translation (OCR)
- 🔲 Plugin system
- 🔲 Mobile app support

## 📞 Support

- **GitHub Issues**: [Report bugs and request features](https://github.com/your-username/live-translation-ai/issues)
- **Discussions**: [Community discussions](https://github.com/your-username/live-translation-ai/discussions)
- **Wiki**: [Additional documentation](https://github.com/your-username/live-translation-ai/wiki)

## 📈 Stats

![GitHub stars](https://img.shields.io/github/stars/your-username/live-translation-ai)
![GitHub forks](https://img.shields.io/github/forks/your-username/live-translation-ai)
![GitHub issues](https://img.shields.io/github/issues/your-username/live-translation-ai)
![GitHub license](https://img.shields.io/github/license/your-username/live-translation-ai)
![Python version](https://img.shields.io/badge/python-3.11+-blue.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

---

**Built with ❤️ for privacy-conscious users who need reliable local translation.**
