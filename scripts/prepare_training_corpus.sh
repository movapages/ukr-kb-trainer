#!/bin/bash
# Prepare training corpus by concatenating all clean text files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INPUT_DIR="$PROJECT_ROOT/stamina/clean"
OUTPUT_FILE="$PROJECT_ROOT/data/training_corpus.txt"

echo "🔄 Preparing training corpus..."
echo "   Input: $INPUT_DIR"
echo "   Output: $OUTPUT_FILE"

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Concatenate all numbered text files (01.txt through 50.txt)
cat "$INPUT_DIR"/{01..50}.txt > "$OUTPUT_FILE" 2>/dev/null

# Check if corpus was created
if [ -f "$OUTPUT_FILE" ]; then
    WORD_COUNT=$(wc -w < "$OUTPUT_FILE")
    CHAR_COUNT=$(wc -c < "$OUTPUT_FILE")
    echo "✅ Corpus created successfully"
    echo "   Words: $WORD_COUNT"
    echo "   Characters: $CHAR_COUNT"
else
    echo "❌ Failed to create corpus"
    exit 1
fi

