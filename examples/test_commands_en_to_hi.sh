#!/bin/bash

# English to Hindi Translation Test Commands
# Simple one-liner commands to test the Live Translation AI system

echo "🇮🇳 English to Hindi Translation Test Commands 🇬🇧"
echo "=================================================="
echo ""

# Make sure we're in the right directory
cd "$(dirname "$0")/.." || exit 1

echo "📍 Working directory: $(pwd)"
echo ""

# Check if the translate command is available
if ! command -v translate &> /dev/null; then
    echo "❌ 'translate' command not found. Please install the package first:"
    echo "   poetry install && poetry shell"
    echo "   # or"
    echo "   pip install -e ."
    echo ""
    exit 1
fi

echo "✅ 'translate' command found"
echo ""

# Test basic text translations
echo "🔄 Running English to Hindi translation tests..."
echo ""

# Test 1: Basic greeting
echo "Test 1: Basic greeting"
translate text "Hello, how are you?" --source en --target hi
echo ""

# Test 2: Thank you
echo "Test 2: Thank you"
translate text "Thank you very much" --source en --target hi
echo ""

# Test 3: Introduction
echo "Test 3: Introduction"
translate text "Nice to meet you, my name is John" --source en --target hi
echo ""

# Test 4: Question
echo "Test 4: Question"
translate text "What is your name?" --source en --target hi
echo ""

# Test 5: Food compliment
echo "Test 5: Food compliment"
translate text "The food is very delicious" --source en --target hi
echo ""

# Test 6: Help request
echo "Test 6: Help request"
translate text "Can you help me please?" --source en --target hi
echo ""

# Test 7: Location question
echo "Test 7: Location question"
translate text "Where is the nearest restaurant?" --source en --target hi
echo ""

# Test 8: Time question
echo "Test 8: Time question"
translate text "What time is it now?" --source en --target hi
echo ""

# Test 9: Shopping
echo "Test 9: Shopping"
translate text "How much does this cost?" --source en --target hi
echo ""

# Test 10: Goodbye
echo "Test 10: Goodbye"
translate text "Thank you and goodbye!" --source en --target hi
echo ""

echo "✅ All translation tests completed!"
echo ""

# Show some useful one-liner commands
echo "💡 Useful one-liner commands you can try:"
echo ""
echo "# Basic translation:"
echo 'translate text "Hello world" --source en --target hi'
echo ""
echo "# Interactive mode:"
echo 'translate text --interactive --source en --target hi'
echo ""
echo "# Translate from stdin:"
echo 'echo "Welcome to India" | translate text --target hi'
echo ""
echo "# Live audio translation (10 seconds):"
echo 'translate audio --source en --target hi --duration 10'
echo ""
echo "# Check available audio devices:"
echo 'translate devices'
echo ""
echo "# System status:"
echo 'translate status'
echo ""

echo "🎉 Happy translating! 🎉"
