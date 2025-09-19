#!/usr/bin/env python3
"""
Example script demonstrating Live Translation AI usage.

This script shows how to use the Live Translation AI library programmatically.
"""

import asyncio
import time

from live_translation.core.models import LanguageCode, TranslationRequest
from live_translation.translation.engine import TranslationPipeline
from live_translation.translation.text_translator import SimpleTranslator


async def main() -> None:
    """Main example function."""
    print("🚀 Live Translation AI - Example Script")
    print("=" * 50)

    # Initialize the translation system
    print("📚 Initializing translation system...")

    # For this example, we'll use the simple translator
    # In a real application, you'd use WhisperAdapter and TransformersTranslator
    translator = SimpleTranslator()
    translator.load()

    pipeline = TranslationPipeline(
        stt_engine=translator,
        translation_engine=translator,
    )

    print("✅ Translation system initialized!")
    print()

    # Example 1: Text translation
    print("📝 Example 1: Text Translation")
    print("-" * 30)

    texts_to_translate = [
        ("Hello, world!", "en", "es"),
        ("How are you today?", "en", "fr"),
        ("Good morning!", "en", "de"),
        ("Thank you very much!", "en", "it"),
    ]

    for text, source, target in texts_to_translate:
        request = TranslationRequest(
            text=text,
            source_language=source,
            target_language=target,
        )

        start_time = time.time()
        response = pipeline.process_request(request)
        duration = (time.time() - start_time) * 1000

        print(f"🗣️  Original ({response.detected_language}): {response.original_text}")
        print(f"🌍 Translated: {response.translated_text}")
        print(f"⚡ Processing time: {duration:.1f}ms")
        print()

    # Example 2: Simulated audio translation
    print("🎤 Example 2: Simulated Audio Translation")
    print("-" * 40)

    # Generate some fake audio data (in real usage, this would come from microphone)
    import numpy as np

    # Generate 2 seconds of sine wave
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    audio_bytes = (audio_signal * 32767).astype(np.int16).tobytes()

    audio_request = TranslationRequest(
        audio_data=audio_bytes,
        sample_rate=sample_rate,
        source_language=LanguageCode.AUTO,
        target_language=LanguageCode.SPANISH,
    )

    start_time = time.time()
    audio_response = pipeline.process_request(audio_request)
    duration = (time.time() - start_time) * 1000

    print(f"🎵 Audio processed: {len(audio_bytes)} bytes at {sample_rate}Hz")
    print(
        f"🗣️  Transcribed ({audio_response.detected_language}): {audio_response.original_text}"
    )
    print(f"🌍 Translated: {audio_response.translated_text}")
    print(f"⚡ Processing time: {duration:.1f}ms")
    print(f"🎯 Confidence: {audio_response.confidence:.3f}")
    print()

    # Example 3: Batch processing
    print("📦 Example 3: Batch Processing")
    print("-" * 30)

    batch_texts = [
        "Good morning!",
        "How can I help you?",
        "Thank you for your time.",
        "Have a great day!",
        "See you later!",
    ]

    print("Processing batch of texts...")
    batch_start = time.time()

    for i, text in enumerate(batch_texts, 1):
        request = TranslationRequest(
            text=text,
            source_language=LanguageCode.ENGLISH,
            target_language=LanguageCode.FRENCH,
        )

        response = pipeline.process_request(request)
        print(f"{i}/5: {text} → {response.translated_text}")

    batch_duration = (time.time() - batch_start) * 1000
    print(f"📊 Batch completed in {batch_duration:.1f}ms ({len(batch_texts)} items)")
    print(f"📈 Average per item: {batch_duration/len(batch_texts):.1f}ms")
    print()

    # Example 4: Error handling
    print("⚠️  Example 4: Error Handling")
    print("-" * 28)

    try:
        # Test with empty text (should be caught by Pydantic validation)
        empty_request = TranslationRequest.__new__(TranslationRequest)
        empty_request.text = ""
        empty_request.audio_data = None
        empty_request.source_language = LanguageCode.ENGLISH
        empty_request.target_language = LanguageCode.SPANISH

        pipeline.process_request(empty_request)

    except Exception as e:
        print(f"✅ Properly caught error: {type(e).__name__}: {e}")

    print()
    print("🎉 Examples completed successfully!")
    print("💡 Try running the CLI commands:")
    print("   translate text 'Hello, world!' --target es")
    print("   translate audio --duration 10")
    print("   translate config")


if __name__ == "__main__":
    # Handle numpy import for example
    try:
        pass
    except ImportError:
        print("❌ NumPy not available - skipping audio example")
        print("Install with: pip install numpy")
        exit(1)

    asyncio.run(main())
