"""Command Line Interface for Live Translation AI."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from live_translation.audio.capture import AudioCapture
from live_translation.audio.processor import BatchAudioProcessor, StreamingTranslator
from live_translation.core.config import settings
from live_translation.core.models import (
    LanguageCode,
    TranslationRequest,
    TranslationResponse,
)
from live_translation.translation.engine import TranslationPipeline
from live_translation.translation.text_translator import TransformersTranslator
from live_translation.translation.whisper_engine import WhisperAdapter
from live_translation.utils.logger import get_logger
from live_translation.voice_cloning import (
    TranslationWithVoiceCloning,
    VoiceCloningTranslationPipeline,
    XTTSVoiceCloner,
)

logger = get_logger(__name__)
console = Console()


class CLIContext:
    """Shared context for CLI commands."""

    def __init__(self) -> None:
        self.translation_pipeline: TranslationPipeline | None = None
        self.voice_cloning_pipeline: VoiceCloningTranslationPipeline | None = None
        self.streaming_translator: StreamingTranslator | None = None

    def get_translation_pipeline(self) -> TranslationPipeline:
        """Get or create translation pipeline."""
        if self.translation_pipeline is None:
            console.print("🔄 Initializing translation models...", style="blue")

            with console.status("[bold blue]Loading models..."):
                # Initialize speech-to-text engine
                try:
                    stt_engine = WhisperAdapter(
                        model_name=settings.whisper_model,
                        device=settings.device,
                    )
                    stt_engine.load()
                    console.print("✅ Whisper model loaded successfully", style="green")
                except Exception as e:
                    console.print(f"❌ Failed to load Whisper model: {e}", style="red")
                    console.print(
                        "💡 Please install: pip install openai-whisper", style="yellow"
                    )
                    raise RuntimeError(
                        "Speech-to-text model required but not available"
                    ) from e

                # Initialize translation engine
                try:
                    translation_engine = TransformersTranslator(
                        model_name=settings.translation_model,
                        device=settings.device,
                    )
                    translation_engine.load()
                    console.print(
                        "✅ Translation model loaded successfully", style="green"
                    )
                except Exception as e:
                    console.print(
                        f"❌ Failed to load translation model: {e}", style="red"
                    )
                    console.print(
                        "💡 Please install: pip install transformers", style="yellow"
                    )
                    raise RuntimeError(
                        "Translation model required but not available"
                    ) from e

                self.translation_pipeline = TranslationPipeline(
                    stt_engine=stt_engine,
                    translation_engine=translation_engine,
                )

            console.print("✅ Models loaded successfully!", style="green")

        return self.translation_pipeline

    def get_voice_cloning_pipeline(
        self, reference_audio_path: str | None = None
    ) -> VoiceCloningTranslationPipeline:
        """Get or create voice cloning pipeline."""
        # Always get the base translation pipeline first
        base_pipeline = self.get_translation_pipeline()

        if self.voice_cloning_pipeline is None or reference_audio_path:
            console.print("🎭 Initializing voice cloning...", style="blue")

            with console.status("[bold blue]Loading voice cloning model..."):
                try:
                    voice_engine = (
                        XTTSVoiceCloner(
                            model_name=settings.xtts_model,
                            device=settings.device,
                        )
                        if settings.voice_cloning_enabled
                        else None
                    )

                    self.voice_cloning_pipeline = VoiceCloningTranslationPipeline(
                        translation_pipeline=base_pipeline,
                        voice_cloning_engine=voice_engine,
                        reference_audio_path=reference_audio_path,
                    )

                    if voice_engine:
                        console.print(
                            "✅ Voice cloning model loaded successfully", style="green"
                        )
                    else:
                        console.print(
                            "💡 Voice cloning disabled in settings", style="yellow"
                        )

                except Exception as e:
                    console.print(f"❌ Failed to load voice cloning: {e}", style="red")
                    console.print("💡 Please install: pip install TTS", style="yellow")
                    # Fall back to regular translation pipeline
                    self.voice_cloning_pipeline = VoiceCloningTranslationPipeline(
                        translation_pipeline=base_pipeline,
                        voice_cloning_engine=None,
                    )

        return self.voice_cloning_pipeline


# Global CLI context
cli_context = CLIContext()


@click.group()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    help="Device to use for models",
)
@click.version_option(version="0.1.0")
def cli(config: Path | None, verbose: bool, device: str | None) -> None:
    """Live Translation AI - Privacy-first local translation tool."""
    if verbose:
        settings.log_level = "DEBUG"

    if device:
        # Type ignore for now - this would need proper validation
        settings.device = device  # type: ignore

    if config:
        console.print(f"📝 Using config file: {config}")


cli_context = CLIContext()


def language_options(func: Callable[..., Any]) -> Callable[..., Any]:
    """Common language options decorator."""
    func = click.option(
        "--target",
        "-t",
        type=click.Choice([lang.value for lang in LanguageCode]),
        default="en",
        help="Target language",
    )(func)
    func = click.option(
        "--source",
        "-s",
        type=click.Choice([lang.value for lang in LanguageCode]),
        default="auto",
        help="Source language (auto-detect by default)",
    )(func)
    return func


def display_translation_result(response: TranslationResponse) -> None:
    """Display translation result in a formatted panel."""
    console.print(
        Panel(
            f"🗣️  Original ({response.detected_language}): {response.original_text}\n"
            f"🌍 Translated: {response.translated_text}\n"
            f"⚡ Time: {response.processing_time_ms:.1f}ms",
            border_style="green",
        )
    )


def display_voice_cloning_result(
    response: TranslationWithVoiceCloning, save_audio: Path | None = None
) -> None:
    """Display translation result with voice cloning in a formatted panel."""
    voice_status = (
        "🎭 Voice cloned" if response.has_voice_cloning else "❌ Voice cloning failed"
    )

    content = (
        f"🗣️  Original ({response.detected_language}): {response.original_text}\n"
        f"🌍 Translated: {response.translated_text}\n"
        f"{voice_status}\n"
        f"⚡ Total time: {response.total_processing_time_ms:.1f}ms"
    )

    if response.has_voice_cloning and response.voice_cloning_response:
        duration = response.voice_cloning_response.duration_seconds
        content += f"\n🎵 Audio duration: {duration:.1f}s"

        # Save audio if requested
        if save_audio and response.cloned_audio_data:
            with open(save_audio, "wb") as f:
                f.write(response.cloned_audio_data)
            content += f"\n💾 Audio saved: {save_audio}"

    console.print(Panel(content, border_style="green"))


def _process_translation_request(
    pipeline: Any,
    request: TranslationRequest,
    use_voice_cloning: bool,
    save_audio: Path | None = None,
) -> None:
    """Process a translation request and display results."""
    with console.status("[bold blue]Translating..."):
        response = pipeline.process_request(request)
        if use_voice_cloning and isinstance(response, TranslationWithVoiceCloning):
            display_voice_cloning_result(response, save_audio)
        elif isinstance(response, TranslationResponse):
            display_translation_result(response)
        else:
            # Fallback for unexpected response types
            console.print(f"❌ Unexpected response type: {type(response)}", style="red")


@cli.command()
@language_options
@click.option("--device", type=int, help="Audio device index")
@click.option(
    "--duration", "-d", type=int, default=30, help="Recording duration in seconds"
)
def audio(source: str, target: str, device: int | None, duration: int) -> None:
    """Translate live audio input."""
    pipeline = cli_context.get_translation_pipeline()

    console.print(
        Panel(
            f"🎤 Live Audio Translation\n"
            f"Source: {source} → Target: {target}\n"
            f"Duration: {duration}s",
            title="Audio Translation",
            border_style="blue",
        )
    )

    # List available devices
    if device is None:
        devices = AudioCapture.list_audio_devices()
        if devices:
            table = Table(title="Available Audio Devices")
            table.add_column("Index", style="cyan")
            table.add_column("Name", style="white")
            table.add_column("Channels", style="green")

            for dev in devices:
                table.add_row(str(dev["index"]), dev["name"], str(dev["channels"]))
            console.print(table)

            device = click.prompt("Select device index", type=int, default=0)

    async def run_streaming() -> None:
        streaming_translator = StreamingTranslator(
            pipeline,
            source_language=source,
            target_language=target,
        )

        def on_translation(response: TranslationResponse) -> None:
            console.print(
                Panel(
                    f"🗣️  Original: {response.original_text}\n"
                    f"🌍 Translated: {response.translated_text}\n"
                    f"⚡ Time: {response.processing_time_ms:.1f}ms",
                    border_style="green",
                )
            )

        try:
            console.print("🎙️  Recording... Press Ctrl+C to stop")

            # Start streaming with timeout
            task = asyncio.create_task(
                streaming_translator.start_streaming(on_translation, device)
            )

            # Wait for duration or keyboard interrupt
            await asyncio.wait_for(task, timeout=duration)

        except TimeoutError:
            console.print(f"⏰ Recording completed after {duration} seconds")
        except KeyboardInterrupt:
            console.print("\n🛑 Recording stopped by user")
        finally:
            await streaming_translator.stop_streaming()

    try:
        asyncio.run(run_streaming())
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        sys.exit(1)


@cli.command()
@language_options
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
@click.option(
    "--voice-clone",
    type=click.Path(exists=True, path_type=Path),
    help="Reference audio file for voice cloning",
)
@click.option(
    "--save-audio",
    type=click.Path(path_type=Path),
    help="Save cloned voice audio to file",
)
@click.argument("text", required=False)
def text(
    source: str,
    target: str,
    interactive: bool,
    voice_clone: Path | None,
    save_audio: Path | None,
    text: str | None,
) -> None:
    """Translate text input with optional voice cloning."""
    # Determine which pipeline to use
    if voice_clone:
        pipeline = cli_context.get_voice_cloning_pipeline(str(voice_clone))
        use_voice_cloning = True
    else:
        pipeline = cli_context.get_translation_pipeline()
        use_voice_cloning = False

    if interactive:
        console.print(
            Panel(
                f"📝 Interactive Text Translation\n"
                f"Source: {source} → Target: {target}\n"
                f"Voice cloning: {'✅ Enabled' if use_voice_cloning else '❌ Disabled'}\n"
                f"Type 'quit' to exit",
                title="Text Translation",
                border_style="blue",
            )
        )

        while True:
            try:
                user_input = click.prompt("Enter text", type=str)
                if user_input.lower() in ["quit", "exit", "q"]:
                    break

                # Translate text
                request = TranslationRequest(
                    text=user_input,
                    source_language=source,
                    target_language=target,
                )

                _process_translation_request(
                    pipeline, request, use_voice_cloning, save_audio
                )

            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"❌ Error: {e}", style="red")

    elif text:
        # Single translation
        try:
            request = TranslationRequest(
                text=text,
                source_language=source,
                target_language=target,
            )

            _process_translation_request(
                pipeline, request, use_voice_cloning, save_audio
            )

        except Exception as e:
            console.print(f"❌ Error: {e}", style="red")
            sys.exit(1)

    else:
        # Read from stdin
        try:
            stdin_text = sys.stdin.read().strip()
            if stdin_text:
                request = TranslationRequest(
                    text=stdin_text,
                    source_language=source,
                    target_language=target,
                )

                with console.status("[bold blue]Translating..."):
                    if use_voice_cloning:
                        response = pipeline.process_request(request)
                        if isinstance(response, TranslationWithVoiceCloning):
                            if (
                                response.has_voice_cloning
                                and save_audio
                                and response.cloned_audio_data
                            ):
                                with open(save_audio, "wb") as f:
                                    f.write(response.cloned_audio_data)
                        # Simple output for piping
                        print(response.translated_text)
                    else:
                        response = pipeline.process_request(request)
                        print(response.translated_text)

        except Exception as e:
            console.print(f"❌ Error: {e}", style="red")
            sys.exit(1)


@cli.command("voice-clone")
@language_options
@click.option(
    "--reference",
    "-r",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Reference audio file for voice cloning",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file for cloned voice audio",
)
@click.option("--speed", type=float, default=1.0, help="Speech speed (0.1-3.0)")
@click.option(
    "--temperature", type=float, default=0.75, help="Model temperature (0.1-1.0)"
)
@click.argument("text", required=True)
def voice_clone_command(
    source: str,
    target: str,
    reference: Path,
    output: Path | None,
    speed: float,
    temperature: float,
    text: str,
) -> None:
    """Clone voice and synthesize translated text."""
    console.print(
        Panel(
            f"🎭 Voice Cloning Translation\n"
            f"Text: {text[:50]}{'...' if len(text) > 50 else ''}\n"
            f"Source: {source} → Target: {target}\n"
            f"Reference: {reference.name}\n"
            f"Speed: {speed}x, Temperature: {temperature}",
            title="Voice Cloning",
            border_style="magenta",
        )
    )

    # Initialize voice cloning pipeline
    pipeline = cli_context.get_voice_cloning_pipeline(str(reference))

    if not pipeline.is_voice_cloning_available():
        console.print("❌ Voice cloning not available", style="red")
        console.print(
            "💡 Enable voice cloning in settings and install TTS", style="yellow"
        )
        sys.exit(1)

    try:
        # Create translation request
        request = TranslationRequest(
            text=text,
            source_language=source,
            target_language=target,
        )

        with console.status("[bold magenta]Cloning voice..."):
            # Update settings for this request
            settings.voice_cloning_speed = speed
            settings.voice_cloning_temperature = temperature

            response = pipeline.process_request(request, enable_voice_cloning=True)

        if isinstance(response, TranslationWithVoiceCloning):
            display_voice_cloning_result(response, output)

            if response.has_voice_cloning:
                console.print("✅ Voice cloning completed successfully!", style="green")

                # Show supported languages
                supported = pipeline.get_supported_voice_languages()
                if supported:
                    console.print(
                        f"💡 Supported languages: {', '.join(supported)}", style="blue"
                    )
            else:
                console.print("❌ Voice cloning failed", style="red")
        else:
            console.print("❌ Voice cloning pipeline error", style="red")
            sys.exit(1)

    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        sys.exit(1)


@cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--source",
    "-s",
    type=click.Choice([lang.value for lang in LanguageCode]),
    default="auto",
    help="Source language",
)
@click.option(
    "--target",
    "-t",
    type=click.Choice([lang.value for lang in LanguageCode]),
    default="en",
    help="Target language",
)
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), help="Output directory"
)
def file(
    files: tuple[Path, ...], source: str, target: str, output: Path | None
) -> None:
    """Translate audio files."""
    if not files:
        console.print("❌ No files provided", style="red")
        sys.exit(1)

    pipeline = cli_context.get_translation_pipeline()
    batch_processor = BatchAudioProcessor(pipeline)

    # Setup output directory
    if output:
        output.mkdir(parents=True, exist_ok=True)
    else:
        output = Path.cwd() / "translations"
        output.mkdir(exist_ok=True)

    async def process_files() -> None:
        file_paths = [str(f) for f in files]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Processing {len(files)} files...", total=len(files)
            )
            completed = 0

            async for file_path, response in batch_processor.process_files(
                file_paths, source, target
            ):
                # Save translation to file
                input_path = Path(file_path)
                output_file = output / f"{input_path.stem}_translation.txt"

                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(
                        f"Original ({response.detected_language}): {response.original_text}\n"
                    )
                    f.write(f"Translated: {response.translated_text}\n")
                    f.write(f"Processing time: {response.processing_time_ms:.1f}ms\n")

                completed += 1
                progress.update(
                    task,
                    advance=1,
                    description=f"Completed {completed}/{len(files)} files",
                )

                console.print(f"✅ Translated: {input_path.name} → {output_file.name}")

    try:
        asyncio.run(process_files())
        console.print(
            f"🎉 All files processed! Results saved in: {output}", style="green"
        )
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        sys.exit(1)


@cli.command()
def devices() -> None:
    """List available audio devices."""
    devices = AudioCapture.list_audio_devices()

    if not devices:
        console.print("❌ No audio devices found", style="red")
        return

    table = Table(title="Available Audio Input Devices")
    table.add_column("Index", style="cyan", justify="center")
    table.add_column("Name", style="white")
    table.add_column("Channels", style="green", justify="center")
    table.add_column("Sample Rate", style="blue", justify="center")

    for device in devices:
        table.add_row(
            str(device["index"]),
            device["name"],
            str(device["channels"]),
            f"{device['sample_rate']:.0f} Hz",
        )

    console.print(table)


@cli.command()
def config() -> None:
    """Show current configuration."""
    table = Table(title="Live Translation AI Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    table.add_column("Description", style="dim")

    # Model settings
    table.add_row("Whisper Model", settings.whisper_model, "Speech-to-text model")
    table.add_row(
        "Translation Model", settings.translation_model, "Text translation model"
    )
    table.add_row("Device", settings.device, "Processing device")

    # Audio settings
    table.add_row("Sample Rate", f"{settings.sample_rate} Hz", "Audio sample rate")
    table.add_row("Chunk Length", f"{settings.chunk_length}s", "Audio chunk duration")

    # Language settings
    table.add_row(
        "Default Source", settings.default_source_lang, "Default source language"
    )
    table.add_row(
        "Default Target", settings.default_target_lang, "Default target language"
    )

    # Voice cloning settings
    table.add_row(
        "Voice Cloning",
        "✅ Enabled" if settings.voice_cloning_enabled else "❌ Disabled",
        "Voice cloning functionality",
    )
    if settings.voice_cloning_enabled:
        table.add_row("XTTS Model", settings.xtts_model, "Voice cloning model")
        table.add_row(
            "Voice Speed", f"{settings.voice_cloning_speed}x", "Default speech speed"
        )
        table.add_row(
            "Voice Temperature",
            f"{settings.voice_cloning_temperature}",
            "Model temperature",
        )

    # Cache settings
    table.add_row("Cache Dir", str(settings.cache_dir), "Cache directory")
    table.add_row("Model Cache", str(settings.model_cache_dir), "Model cache directory")

    console.print(table)


@cli.command()
def status() -> None:
    """Show system status and model information."""
    console.print(
        Panel(
            "🔍 Checking system status...", title="System Status", border_style="blue"
        )
    )

    # Check model availability
    status_table = Table(title="Model Status")
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", style="white")
    status_table.add_column("Details", style="dim")

    # Check Whisper
    try:
        pass

        whisper_status = "✅ Available"
        whisper_details = f"Model: {settings.whisper_model}"
    except ImportError:
        whisper_status = "❌ Not installed"
        whisper_details = "pip install openai-whisper"

    status_table.add_row("Whisper STT", whisper_status, whisper_details)

    # Check Transformers
    try:
        transformers_status = "✅ Available"
        transformers_details = f"PyTorch: {torch.__version__}"
    except ImportError:
        transformers_status = "❌ Not installed"
        transformers_details = "pip install transformers torch"

    status_table.add_row("Transformers", transformers_status, transformers_details)

    # Check Audio
    try:
        pass

        audio_status = "✅ Available"
        audio_details = f"Devices: {len(AudioCapture.list_audio_devices())}"
    except ImportError:
        audio_status = "❌ Not installed"
        audio_details = "pip install sounddevice"

    status_table.add_row("Audio Capture", audio_status, audio_details)

    # Check Voice Cloning
    try:
        pass

        voice_status = "✅ Available"
        voice_details = f"XTTS: {settings.xtts_model}"
        if not settings.voice_cloning_enabled:
            voice_status += " (Disabled)"
    except ImportError:
        voice_status = "❌ Not installed"
        voice_details = "pip install TTS"

    status_table.add_row("Voice Cloning", voice_status, voice_details)

    console.print(status_table)


def main() -> None:
    """Main CLI entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        console.print(f"💥 Unexpected error: {e}", style="red")
        sys.exit(1)


if __name__ == "__main__":
    main()
