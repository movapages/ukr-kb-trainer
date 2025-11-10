# 🇺🇦 Model Training Guide

How to train the Ukrainian language model for the keyboard trainer.

---

## Quick Start

### 1. Prepare Training Data

Create a text file with Ukrainian content:

```bash
# Option A: Use existing word pool
cat data/word_pool/*.txt > corpus.txt

# Option B: Use any Ukrainian text corpus
# Download from: https://huggingface.co/datasets/wmt14/wmt14
```

### 2. Train the Model

```bash
cargo run --bin train -- \
  --corpus corpus.txt \
  --output models/ \
  --epochs 10 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --context 32 \
  --embedding-dim 64 \
  --hidden-dim 128 \
  --verbose
```

### 3. Run the Trainer

```bash
# After training, run the interactive drill
cargo run --release --bin ukr-kb-trainer -- --level 1 --debug
```

**You should see:**
```
🇺🇦 Ukrainian Keyboard Trainer v0.1.0
Loading 51 words for Level 1 (showing up to 50)...

Target: аа
Your Input: а_

Progress: 1/50 words  |  Accuracy: 50%  |  Next check: 10 words
```

---

## Training Options

### Basic Training (Minimal)
```bash
cargo run --bin train -- --corpus corpus.txt
```

### Production Training (Full GPU)
```bash
cargo run --bin train -- \
  --corpus corpus.txt \
  --output models/ \
  --epochs 50 \
  --batch-size 64 \
  --learning-rate 0.0001 \
  --context 64 \
  --embedding-dim 256 \
  --hidden-dim 512 \
  --gpu \
  --verbose
```

### Fine-tuning on Existing Model
```bash
cargo run --bin train -- \
  --corpus corpus.txt \
  --load models/model_weights.bin \
  --epochs 5 \
  --learning-rate 0.00001
```

---

## Output Files

After training, you'll have:

```
models/
├── model_weights.bin    ← Model weights (bincode format)
└── vocab.json           ← Character vocabulary mapping
```

---

## Model Architecture

### Character-Level LLM

```
Input Text
    ↓
Character Embedding (64-dim)
    ↓
LSTM Layer 1 (128-hidden)
    ↓
LSTM Layer 2 (128-hidden)
    ↓
Output Linear (vocab_size)
    ↓
Next Character Prediction
```

**Parameters:**
- Vocab size: ~100 (Ukrainian alphabet + symbols)
- Embedding dim: 64
- Hidden dim: 128
- Total params: ~45K (very small model)

---

## Training Data Sources

### 1. Word Pool (Already Available)
```bash
cat data/word_pool/*.txt > corpus.txt  # ~5K words
```

### 2. Ukrainian Wikipedia Dump
- Download: https://dumps.wikimedia.org/ukwiki/
- Extract and clean Ukrainian text

### 3. HuggingFace Datasets
```python
from datasets import load_dataset
ds = load_dataset("wmt14", "de-en")  # or other Ukrainian corpus
```

### 4. Ukrainian News Corpora
- Ukrainian News Corpus: https://github.com/orimraf/Ukrainian-News-Corpus

---

## Metrics & Monitoring

**Training Metrics:**
- Loss: Cross-entropy loss (lower is better)
- Perplexity: exp(loss)
- Character accuracy: % correct next-char predictions

**Monitoring:**
```bash
# Tail training progress
cargo run --bin train -- --corpus corpus.txt --verbose | tail -20
```

---

## Tips for Better Training

### 1. Data Quality
- Clean text (remove non-Ukrainian characters)
- Normalize apostrophes: `ʼ` → `'`
- Consistent capitalization

### 2. Hyperparameters
- Start small: 32 context, 64 embedding
- Increase gradually: 64 context, 128 embedding
- Use learning rate schedule: decay over epochs

### 3. GPU Acceleration
```bash
# Enable GPU if available
cargo run --bin train -- --corpus corpus.txt --gpu
```

### 4. Checkpoint & Resume
- Save best model during training
- Resume from checkpoint on interruption

---

## Validation & Testing

### Quick Validation
```bash
# Check model loads correctly
cargo run --release --bin ukr-kb-trainer -- --level 1 --debug

# Should output:
# Loading 51 words for Level 1 (showing up to 50)...
# ✓ Vocabulary loaded: 101 characters
# Target: аа (or other word from pool)
# Progress: 1/50 words  |  Accuracy: 0%  |  Next check: 10 words
```

### Full Session Test
```bash
cargo run -- --level 5
# Type: "мама" (should show ~accuracy tracking)
```

---

## Production Deployment

### 1. Export Model
```bash
cp models/model_weights.bin /path/to/release/
cp models/vocab.json /path/to/release/
```

### 2. Build Release
```bash
cargo build --release
```

### 3. Run Production
```bash
./target/release/ukr-kb-trainer --level 1 \
  --model models/model_weights.bin \
  --vocab models/vocab.json
```

---

## Troubleshooting

### Issue: Training is slow
**Solution:** Reduce vocab size or use GPU with `--gpu`

### Issue: Model weights file is huge
**Solution:** Use compression: `gzip -9 models/model_weights.bin`

### Issue: Out of memory
**Solution:** Reduce `--batch-size` or `--context`

### Issue: Poor accuracy after training
**Solution:** 
- Train longer (increase `--epochs`)
- Use better data (quality Ukrainian text)
- Increase model size (`--hidden-dim 256`)

---

## Next Steps

### Phase 6B: Real Model Training
1. ✅ Create training pipeline (this file)
2. ⏭️  Train on word pool data
3. ⏭️  Validate model loading
4. ⏭️  Test full trainer app

### Phase 6C: Production Model (Optional)
1. Train on full Ukrainian corpus (~100M characters)
2. Implement multi-layer LSTM
3. Add attention mechanism
4. Fine-tune on typing patterns

---

## References

- Candle Documentation: https://huggingface.co/docs/candle/
- Ukrainian NLP: https://github.com/eternalviolet/Ukrainian-NLP-resources
- Character-Level Models: https://arxiv.org/abs/1508.06615

---

**Version:** 0.1.0  
**Status:** Phase 6B - Training Pipeline Ready

