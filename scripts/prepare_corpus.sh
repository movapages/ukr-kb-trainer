#!/bin/bash
# Prepare Ukrainian text corpus for training
# Usage: ./scripts/prepare_corpus.sh [output_file]

set -e

OUTPUT="${1:-corpus.txt}"
WORK_DIR=$(mktemp -d)
trap "rm -rf $WORK_DIR" EXIT

echo "🇺🇦 Preparing Ukrainian Training Corpus"
echo "========================================"
echo ""

# 1. Combine word pool
echo "📚 Step 1: Combining word pools..."
cat data/word_pool/*.txt > "$WORK_DIR/wordpool.txt"
WORD_COUNT=$(wc -w < "$WORK_DIR/wordpool.txt")
echo "   ✓ Combined $WORD_COUNT words from word_pool/"

# 2. Clean and normalize text
echo ""
echo "🧹 Step 2: Cleaning and normalizing..."
python3 << 'EOF' "$WORK_DIR/wordpool.txt" "$OUTPUT"
import sys
import re
import unicodedata

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Normalize apostrophes
text = text.replace('ʼ', "'")
text = text.replace('`', "'")
text = text.replace(''', "'")

# Remove extra whitespace
text = re.sub(r'\s+', ' ', text)
text = text.strip()

# Write output
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(text)

print(f"   ✓ Normalized text: {len(text)} characters")
EOF

# 3. Summary
echo ""
echo "✅ Corpus prepared!"
FILE_SIZE=$(du -h "$OUTPUT" | cut -f1)
CHAR_COUNT=$(wc -c < "$OUTPUT")
echo ""
echo "📊 Summary:"
echo "   File: $OUTPUT"
echo "   Size: $FILE_SIZE"
echo "   Characters: $CHAR_COUNT"
echo ""
echo "🚀 Next: Train the model with:"
echo "   cargo run --bin train -- --corpus $OUTPUT --epochs 10"

