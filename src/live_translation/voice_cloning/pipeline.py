"""Pipeline adapter integrating voice cloning with translation."""

from __future__ import annotations

import time

from live_translation.core.config import settings
from live_translation.core.models import TranslationRequest, TranslationResponse
from live_translation.translation.engine import TranslationPipeline
from live_translation.utils.logger import get_logger

from .engine import VoiceCloningEngine
from .models import VoiceCloningRequest, VoiceCloningResponse
from .xtts_cloner import XTTSVoiceCloner

logger = get_logger(__name__)


class TranslationWithVoiceCloning:
    """Extended translation response with voice cloning."""

    def __init__(
        self,
        translation_response: TranslationResponse,
        voice_cloning_response: VoiceCloningResponse | None = None,
    ):
        """Initialize translation with voice cloning response.

        Args:
            translation_response: Standard translation response
            voice_cloning_response: Voice cloning response if available
        """
        self.translation_response = translation_response
        self.voice_cloning_response = voice_cloning_response

    @property
    def original_text(self) -> str:
        """Get original text."""
        return self.translation_response.original_text

    @property
    def translated_text(self) -> str:
        """Get translated text."""
        return self.translation_response.translated_text

    @property
    def detected_language(self) -> str:
        """Get detected language."""
        return self.translation_response.detected_language or "unknown"

    @property
    def cloned_audio_data(self) -> bytes | None:
        """Get cloned voice audio data."""
        return (
            self.voice_cloning_response.audio_data
            if self.voice_cloning_response
            else None
        )

    @property
    def has_voice_cloning(self) -> bool:
        """Check if voice cloning was performed."""
        return self.voice_cloning_response is not None

    @property
    def total_processing_time_ms(self) -> float:
        """Get total processing time including voice cloning."""
        base_time = self.translation_response.processing_time_ms
        voice_time = (
            self.voice_cloning_response.processing_time_ms
            if self.voice_cloning_response
            else 0
        )
        return base_time + voice_time


class VoiceCloningTranslationPipeline:
    """Translation pipeline with integrated voice cloning."""

    def __init__(
        self,
        translation_pipeline: TranslationPipeline,
        voice_cloning_engine: VoiceCloningEngine | None = None,
        reference_audio_path: str | None = None,
    ):
        """Initialize voice cloning translation pipeline.

        Args:
            translation_pipeline: Standard translation pipeline
            voice_cloning_engine: Voice cloning engine (auto-created if None)
            reference_audio_path: Path to reference voice audio
        """
        self.translation_pipeline = translation_pipeline
        self.reference_audio_path = reference_audio_path
        self.voice_cloning_engine: VoiceCloningEngine | None

        # Initialize voice cloning engine
        if voice_cloning_engine is None and settings.voice_cloning_enabled:
            self.voice_cloning_engine = XTTSVoiceCloner(
                model_name=settings.xtts_model,
                device=settings.device,
            )
        else:
            self.voice_cloning_engine = voice_cloning_engine

        # Pre-load reference voice if available
        if self.voice_cloning_engine and self.reference_audio_path:
            if self.voice_cloning_engine.load_reference_voice(
                self.reference_audio_path
            ):
                logger.info(f"✅ Reference voice loaded: {self.reference_audio_path}")
            else:
                logger.warning(
                    f"⚠️ Failed to load reference voice: {self.reference_audio_path}"
                )

    def process_request(
        self,
        request: TranslationRequest,
        reference_audio_path: str | None = None,
        enable_voice_cloning: bool | None = None,
    ) -> TranslationWithVoiceCloning:
        """Process translation request with optional voice cloning.

        Args:
            request: Translation request
            reference_audio_path: Path to reference audio (overrides default)
            enable_voice_cloning: Enable/disable voice cloning for this request

        Returns:
            TranslationWithVoiceCloning response
        """
        start_time = time.time()

        # Perform standard translation
        logger.debug("Processing translation request...")
        translation_response = self.translation_pipeline.process_request(request)

        # Check if voice cloning should be performed
        should_clone_voice = self._should_perform_voice_cloning(
            enable_voice_cloning, reference_audio_path
        )

        voice_cloning_response = None

        if should_clone_voice:
            try:
                logger.debug("Starting voice cloning...")
                ref_audio = reference_audio_path or self.reference_audio_path
                if ref_audio:
                    voice_cloning_response = self._perform_voice_cloning(
                        translation_response.translated_text,
                        ref_audio,
                        request.target_language,
                    )
                    logger.info("✅ Voice cloning completed successfully")
                else:
                    logger.warning("No reference audio available for voice cloning")

            except Exception as e:
                logger.error(f"❌ Voice cloning failed: {e}")
                # Continue without voice cloning rather than failing the entire request

        total_time = (time.time() - start_time) * 1000
        logger.debug(f"Pipeline completed in {total_time:.1f}ms")

        return TranslationWithVoiceCloning(
            translation_response=translation_response,
            voice_cloning_response=voice_cloning_response,
        )

    def _should_perform_voice_cloning(
        self,
        enable_voice_cloning: bool | None,
        reference_audio_path: str | None,
    ) -> bool:
        """Determine if voice cloning should be performed."""
        # Explicit disable
        if enable_voice_cloning is False:
            return False

        # Check if voice cloning is available
        if not self.voice_cloning_engine:
            return False

        # Check if we have a reference audio path
        ref_path = reference_audio_path or self.reference_audio_path
        if not ref_path:
            logger.debug("No reference audio path available for voice cloning")
            return False

        # Explicit enable or default setting
        return enable_voice_cloning is True or settings.voice_cloning_enabled

    def _perform_voice_cloning(
        self,
        text: str,
        reference_audio_path: str,
        target_language: str,
    ) -> VoiceCloningResponse:
        """Perform voice cloning synthesis."""
        if not self.voice_cloning_engine:
            raise ValueError("Voice cloning engine is not available")

        # Create voice cloning request
        voice_request = VoiceCloningRequest(
            text=text,
            reference_audio_path=reference_audio_path,
            target_language=target_language,
            sample_rate=settings.voice_sample_rate,
            speed=settings.voice_cloning_speed,
            temperature=settings.voice_cloning_temperature,
        )

        # Perform voice cloning
        return self.voice_cloning_engine.clone_voice(voice_request)

    def set_reference_voice(self, audio_path: str) -> bool:
        """Set reference voice for future requests.

        Args:
            audio_path: Path to reference audio file

        Returns:
            True if reference voice was successfully loaded
        """
        if not self.voice_cloning_engine:
            logger.warning("Voice cloning engine not available")
            return False

        if self.voice_cloning_engine.load_reference_voice(audio_path):
            self.reference_audio_path = audio_path
            logger.info(f"✅ Reference voice updated: {audio_path}")
            return True
        else:
            logger.error(f"❌ Failed to load reference voice: {audio_path}")
            return False

    def is_voice_cloning_available(self) -> bool:
        """Check if voice cloning is available."""
        return self.voice_cloning_engine is not None

    def get_supported_voice_languages(self) -> list[str]:
        """Get list of supported languages for voice cloning.

        Returns:
            List of supported language codes
        """
        if self.voice_cloning_engine and hasattr(
            self.voice_cloning_engine, "get_supported_languages"
        ):
            return self.voice_cloning_engine.get_supported_languages()  # type: ignore

        # Create a temporary engine to get supported languages
        temp_engine = XTTSVoiceCloner()
        return temp_engine.get_supported_languages()
