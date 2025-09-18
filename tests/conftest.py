"""Test configuration and fixtures."""

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pytest

from live_translation.core.config import settings
from live_translation.core.models import AudioChunk, TranslationRequest
from live_translation.translation.engine import TranslationPipeline
from live_translation.translation.text_translator import SimpleTranslator


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_audio_data() -> bytes:
    """Generate sample audio data for testing."""
    # Generate 1 second of sine wave at 440 Hz
    sample_rate = 16000
    duration = 1.0
    frequency = 440

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t)

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()


@pytest.fixture
def sample_audio_chunk(sample_audio_data: bytes) -> AudioChunk:
    """Create a sample audio chunk."""
    return AudioChunk(
        data=sample_audio_data,
        sample_rate=16000,
        channels=1,
        duration_ms=1000.0,
        sequence_id=1,
    )


@pytest.fixture
def simple_translator() -> SimpleTranslator:
    """Create a simple translator for testing."""
    return SimpleTranslator()


@pytest.fixture
def translation_pipeline(simple_translator: SimpleTranslator) -> TranslationPipeline:
    """Create a translation pipeline with mock engines."""
    return TranslationPipeline(
        stt_engine=simple_translator,  # type: ignore
        translation_engine=simple_translator,
    )


@pytest.fixture
def sample_translation_request() -> TranslationRequest:
    """Create a sample translation request."""
    return TranslationRequest(
        text="Hello, world!",
        source_language="en",
        target_language="es",
    )


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def setup_test_environment(temp_dir: Path) -> None:
    """Setup test environment with temporary directories."""
    # Override cache directories for tests
    settings.cache_dir = temp_dir / "cache"
    settings.model_cache_dir = temp_dir / "models"
    settings.log_level = "DEBUG"

    # Create directories
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def mock_whisper_model() -> dict[str, Any]:
    """Mock Whisper model for testing."""
    return {
        "transcribe": lambda audio, **kwargs: {
            "text": "Mock transcription",
            "language": "en",
        }
    }


# Pytest configuration
def pytest_configure(config: Any) -> None:
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
