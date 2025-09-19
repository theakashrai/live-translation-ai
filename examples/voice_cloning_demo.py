"""Example script demonstrating voice cloning with translation workflow."""

from __future__ import annotations

import asyncio

from live_translation.core.config import settings
from live_translation.core.models import TranslationRequest
from live_translation.translation.engine import TranslationPipeline
from live_translation.translation.text_translator import TransformersTranslator
from live_translation.translation.whisper_engine import WhisperAdapter
from live_translation.voice_cloning import (
    VoiceCloningTranslationPipeline,
    XTTSVoiceCloner,
)


async def main() -> None:
    """Demonstrate voice cloning with translation."""
    print("🎭 Voice Cloning Translation Demo")
    print("=" * 50)

    # Enable voice cloning in settings
    settings.voice_cloning_enabled = True

    # Initialize translation components
    print("🔄 Loading models...")

    # Speech-to-text engine
    stt_engine = WhisperAdapter(
        model_name=settings.whisper_model,
        device=settings.device,
    )

    # Text translation engine
    translation_engine = TransformersTranslator(
        model_name=settings.translation_model,
        device=settings.device,
    )

    # Standard translation pipeline
    translation_pipeline = TranslationPipeline(
        stt_engine=stt_engine,
        translation_engine=translation_engine,
    )

    # Voice cloning engine
    voice_engine = XTTSVoiceCloner(
        model_name=settings.xtts_model,
        device=settings.device,
    )

    print("✅ Models loaded successfully!")

    # Create voice cloning pipeline
    # Note: In a real scenario, you would provide a valid reference audio path
    reference_audio = "path/to/reference_voice.wav"  # Update this path

    voice_pipeline = VoiceCloningTranslationPipeline(
        translation_pipeline=translation_pipeline,
        voice_cloning_engine=voice_engine,
        reference_audio_path=reference_audio,
    )

    print(f"\n🎤 Reference voice: {reference_audio}")
    print(
        f"🌍 Supported voice languages: {', '.join(voice_pipeline.get_supported_voice_languages())}"
    )

    # Example texts for translation and voice cloning
    example_texts = [
        ("Hello, how are you today?", "en", "es"),
        ("This is a demonstration of voice cloning technology.", "en", "fr"),
        ("The weather is beautiful today.", "en", "de"),
    ]

    for i, (text, source_lang, target_lang) in enumerate(example_texts, 1):
        print(f"\n{'='*20} Example {i} {'='*20}")
        print(f"📝 Original text ({source_lang}): {text}")

        # Create translation request
        request = TranslationRequest(
            text=text,
            source_language=source_lang,
            target_language=target_lang,
        )

        try:
            # Process with voice cloning
            print("⚡ Processing translation and voice cloning...")
            response = voice_pipeline.process_request(request)

            print(f"🌍 Translated ({target_lang}): {response.translated_text}")
            print(
                f"⏱️  Translation time: {response.translation_response.processing_time_ms:.1f}ms"
            )

            if response.has_voice_cloning and response.voice_cloning_response:
                vc_response = response.voice_cloning_response
                print("🎭 Voice cloned: ✅")
                print(f"🎵 Audio duration: {vc_response.duration_seconds:.2f}s")
                print(f"⏱️  Voice cloning time: {vc_response.processing_time_ms:.1f}ms")
                print(f"⚡ Total time: {response.total_processing_time_ms:.1f}ms")

                # Save audio example
                output_path = f"example_output_{i}_{target_lang}.wav"
                with open(output_path, "wb") as f:
                    f.write(vc_response.audio_data)
                print(f"💾 Audio saved: {output_path}")
            else:
                print("🎭 Voice cloning: ❌ Failed or disabled")

        except Exception as e:
            print(f"❌ Error processing example {i}: {e}")

    print(f"\n{'='*50}")
    print("✅ Voice cloning demonstration completed!")
    print("\n💡 Tips:")
    print("- Provide a high-quality reference audio file (10-30 seconds)")
    print("- Ensure the reference audio contains clear speech")
    print("- Supported languages depend on the XTTS model")
    print("- Voice quality improves with better reference audio")


if __name__ == "__main__":
    # Check if voice cloning is available
    try:
        pass

        asyncio.run(main())
    except ImportError:
        print("❌ TTS library not installed!")
        print("💡 Install with: pip install TTS")
    except Exception as e:
        print(f"❌ Error: {e}")
