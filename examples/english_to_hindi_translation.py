#!/usr/bin/env python3
"""
Simple English to Hindi translation example.

This script demonstrates basic English to Hindi translation using the Live Translation AI library.
Run this script to see some example translations.
"""

import asyncio
import time
from typing import List, Tuple

from live_translation.core.models import LanguageCode, TranslationRequest
from live_translation.translation.engine import TranslationPipeline
from live_translation.translation.text_translator import SimpleTranslator


def print_header() -> None:
    """Print a nice header."""
    print("🇮🇳 English to Hindi Translation Examples 🇬🇧")
    print("=" * 50)
    print()


def print_translation(original: str, translated: str, duration: float) -> None:
    """Print a formatted translation result."""
    print(f"🇬🇧 English: {original}")
    print(f"🇮🇳 Hindi:   {translated}")
    print(f"⚡ Time:    {duration:.1f}ms")
    print("-" * 40)


async def main() -> None:
    """Run English to Hindi translation examples."""
    print_header()

    # Initialize the translation system
    print("📚 Initializing translation system...")

    # For this example, we'll use the simple translator
    translator = SimpleTranslator()
    translator.load()

    pipeline = TranslationPipeline(
        stt_engine=translator,
        translation_engine=translator,
    )

    print("✅ Translation system ready!")
    print()

    # Common English phrases to translate to Hindi
    phrases_to_translate: List[Tuple[str, str]] = [
        ("Hello, how are you?", "Basic greeting"),
        ("Good morning!", "Morning greeting"),
        ("Good evening!", "Evening greeting"),
        ("Thank you very much!", "Expressing gratitude"),
        ("Nice to meet you", "Meeting someone new"),
        ("What is your name?", "Asking for name"),
        ("I am learning Hindi", "Language learning"),
        ("How much does this cost?", "Shopping question"),
        ("Where is the bathroom?", "Common travel question"),
        ("I need help", "Asking for assistance"),
        ("The food is delicious", "Compliment about food"),
        ("I am from America", "Nationality statement"),
        ("Can you speak English?", "Language inquiry"),
        ("I don't understand", "Communication difficulty"),
        ("Please speak slowly", "Request for clarity"),
    ]

    print("🔄 Translating common English phrases to Hindi:")
    print()

    total_start_time = time.time()

    for i, (phrase, context) in enumerate(phrases_to_translate, 1):
        print(f"Example {i}/15: {context}")

        request = TranslationRequest(
            text=phrase,
            source_language=LanguageCode.ENGLISH,
            target_language=LanguageCode.HINDI,
        )

        start_time = time.time()
        response = pipeline.process_request(request)
        duration = (time.time() - start_time) * 1000

        print_translation(phrase, response.translated_text, duration)

        # Small delay for readability
        await asyncio.sleep(0.1)

    total_duration = (time.time() - total_start_time) * 1000

    print()
    print("📊 Summary:")
    print(f"✅ Translated {len(phrases_to_translate)} phrases")
    print(f"⏱️  Total time: {total_duration:.1f}ms")
    print(
        f"📈 Average per phrase: {total_duration/len(phrases_to_translate):.1f}ms")
    print()
    print("💡 Try these CLI commands:")
    print('   translate text "Hello, world!" --source en --target hi')
    print('   translate text --interactive --source en --target hi')
    print('   translate audio --source en --target hi --duration 10')
    print()
    print("🎉 English to Hindi translation examples completed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Translation examples stopped by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("💡 Make sure you have installed all dependencies:")
        print("   poetry install")
        print("   # or")
        print("   pip install -e .")
