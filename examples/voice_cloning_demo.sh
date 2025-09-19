#!/bin/bash

# Voice Cloning Translation Demo
# This script demonstrates the voice cloning functionality

set -e

echo "🎭 Voice Cloning Translation Demo"
echo "================================="

# Check if TTS is installed
if ! python -c "import TTS" 2>/dev/null; then
    echo "❌ TTS library not installed!"
    echo "💡 Installing TTS..."
    pip install TTS
fi

# Enable voice cloning in environment
export TRANSLATION_VOICE_CLONING_ENABLED=true

echo "📋 Example commands:"
echo ""

# Example 1: Simple text translation with voice cloning
echo "1. Translate text with voice cloning:"
echo "   translate voice-clone --reference examples/sample_voice.wav \"Hello, how are you?\" --target es --output hello_es.wav"
echo ""

# Example 2: Interactive text with voice cloning
echo "2. Interactive mode with voice cloning:"
echo "   translate text --interactive --voice-clone examples/sample_voice.wav --target fr"
echo ""

# Example 3: Check voice cloning status
echo "3. Check voice cloning status:"
echo "   translate status"
echo ""

# Example 4: Show configuration including voice settings
echo "4. Show configuration:"
echo "   translate config"
echo ""

echo "💡 Note: You need to provide a reference audio file (WAV format, 10-30 seconds)"
echo "   Example reference audio files should be placed in the examples/ directory"
echo ""

# Run the demo script if available
if [ -f "examples/voice_cloning_demo.py" ]; then
    echo "🚀 Running Python demo script..."
    python examples/voice_cloning_demo.py
else
    echo "⚠️  Python demo script not found. Run manually with:"
    echo "   python examples/voice_cloning_demo.py"
fi

echo ""
echo "✅ Demo completed!"
echo ""
echo "📖 For more information:"
echo "   - Check the voice_cloning module documentation"
echo "   - Use 'translate --help' to see all options"
echo "   - Use 'translate voice-clone --help' for voice cloning options"
