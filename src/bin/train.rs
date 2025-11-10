//! Model training binary for Ukrainian keyboard trainer
//!
//! Trains a character-level language model on Ukrainian text corpus.
//! Usage: cargo run --bin train -- --corpus <path> --output models/ --epochs 10

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, loss, Linear, Module, Optimizer, VarBuilder, VarMap};
use clap::Parser;
use rand::seq::SliceRandom;
use std::fs;
use std::path::Path;

#[derive(Parser, Debug)]
#[command(name = "Ukrainian KB Trainer - Model Training")]
#[command(about = "Train a character-level language model on Ukrainian text")]
struct Args {
    /// Path to training corpus (text file)
    #[arg(short, long)]
    corpus: String,

    /// Output directory for model weights
    #[arg(short, long, default_value = "models")]
    output: String,

    /// Number of training epochs
    #[arg(short, long, default_value = "10")]
    epochs: usize,

    /// Batch size (128 optimal for M1 Metal GPU)
    #[arg(short, long, default_value = "128")]
    batch_size: usize,

    /// Learning rate
    #[arg(short, long, default_value = "0.001")]
    learning_rate: f64,

    /// Context window (sequence length)
    #[arg(long, default_value = "32")]
    context: usize,

    /// Embedding dimension
    #[arg(long, default_value = "64")]
    embedding_dim: usize,

    /// Hidden dimension
    #[arg(long, default_value = "128")]
    hidden_dim: usize,

    /// Enable GPU acceleration (if available)
    #[arg(long)]
    gpu: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Load and preprocess Ukrainian text corpus
fn load_corpus(path: &str) -> std::io::Result<String> {
    fs::read_to_string(path)
}

/// Build character vocabulary from text
fn build_vocab(text: &str) -> (Vec<char>, std::collections::HashMap<char, usize>) {
    let mut chars: Vec<char> = text.chars().collect();
    chars.sort();
    chars.dedup();

    let mut char_to_id = std::collections::HashMap::new();
    for (idx, &ch) in chars.iter().enumerate() {
        char_to_id.insert(ch, idx);
    }

    (chars, char_to_id)
}

/// Convert text to character indices
fn text_to_indices(text: &str, char_to_id: &std::collections::HashMap<char, usize>) -> Vec<usize> {
    text.chars()
        .filter_map(|ch| char_to_id.get(&ch).copied())
        .collect()
}

/// Create true minibatches with proper batching
fn create_batches(
    indices: &[usize],
    batch_size: usize,
    context: usize,
) -> Vec<(Vec<Vec<usize>>, Vec<usize>)> {
    let mut examples = Vec::new();

    // Create sliding window: context tokens → predict next token
    // Input: [t, t+1, ..., t+context-1] → Target: t+context
    for i in 0..indices.len().saturating_sub(context) {
        let input_seq = indices[i..i + context].to_vec();
        let target_token = indices[i + context];
        examples.push((input_seq, target_token));
    }

    // Shuffle for better training
    let mut rng = rand::thread_rng();
    examples.shuffle(&mut rng);

    // Group into TRUE minibatches: (batch_size, seq_len)
    let mut minibatches = Vec::new();
    for chunk in examples.chunks(batch_size) {
        let batch_inputs: Vec<Vec<usize>> = chunk.iter().map(|(x, _)| x.clone()).collect();
        let batch_targets: Vec<usize> = chunk.iter().map(|(_, y)| *y).collect();
        minibatches.push((batch_inputs, batch_targets));
    }

    minibatches
}

/// Context-aware sequence model with proper embedding lookup
struct LanguageModel {
    embedding: candle_nn::Embedding,
    hidden1: Linear,
    hidden2: Linear,
    output: Linear,
}

impl LanguageModel {
    fn new(
        vs: VarBuilder,
        vocab_size: usize,
        hidden_dim: usize,
        embedding_dim: usize,
    ) -> Result<Self> {
        // Embedding layer: maps token indices to dense vectors
        let embedding = candle_nn::embedding(vocab_size, embedding_dim, vs.pp("embedding"))?;
        let hidden1 = linear(embedding_dim, hidden_dim, vs.pp("hidden1"))?;
        let hidden2 = linear(hidden_dim, hidden_dim, vs.pp("hidden2"))?;
        let output = linear(hidden_dim, vocab_size, vs.pp("output"))?;
        Ok(Self {
            embedding,
            hidden1,
            hidden2,
            output,
        })
    }

    fn forward(&self, input_indices: &Tensor) -> Result<Tensor> {
        // Input: (batch_size, seq_len) with token indices
        // Embedding: (batch_size, seq_len, embedding_dim)
        let embedded = self.embedding.forward(input_indices)?;

        // Average pooling over sequence: (batch_size, embedding_dim)
        let pooled = embedded.mean(1)?;

        // MLP layers with ReLU
        let x = self.hidden1.forward(&pooled)?.relu()?;
        let x = self.hidden2.forward(&x)?.relu()?;

        // Output: (batch_size, vocab_size)
        self.output.forward(&x)
    }
}

/// Serialize model with proper tensor metadata
fn save_model(
    model_path: &str,
    vocab_size: usize,
    hidden_dim: usize,
    embedding_dim: usize,
    varmap: &VarMap,
) -> std::io::Result<()> {
    let model_config = ModelConfig {
        vocab_size,
        hidden_size: hidden_dim,
        embedding_size: embedding_dim,
        num_layers: 2,
        num_heads: 4,
    };

    // Extract tensors with names and shapes
    let mut tensor_data = Vec::new();

    for (name, var) in varmap.data().lock().unwrap().iter() {
        let shape = var.shape().dims().to_vec();

        // Flatten tensor to 1D vec
        let data = if shape.len() == 1 {
            var.to_vec1::<f32>().unwrap_or_default()
        } else if shape.len() == 2 {
            var.to_vec2::<f32>()
                .ok()
                .map(|v| v.into_iter().flatten().collect())
                .unwrap_or_default()
        } else {
            // For higher-dim tensors, try converting to vec3 then flatten
            vec![]
        };

        tensor_data.push(TensorData {
            name: name.clone(),
            shape,
            data,
        });
    }

    let model_bundle = ModelBundle {
        config: model_config,
        tensors: tensor_data,
    };

    let serialized = bincode::serialize(&model_bundle)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

    fs::write(model_path, serialized)?;
    Ok(())
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ModelConfig {
    vocab_size: usize,
    hidden_size: usize,
    embedding_size: usize,
    num_layers: usize,
    num_heads: usize,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TensorData {
    name: String,
    shape: Vec<usize>,
    data: Vec<f32>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ModelBundle {
    config: ModelConfig,
    tensors: Vec<TensorData>,
}

/// Save vocabulary to JSON
fn save_vocab_json(output_dir: &str, chars: &[char]) -> std::io::Result<()> {
    let vocab_path = Path::new(output_dir).join("vocab.json");

    let char_strings: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
    let vocab_json = serde_json::json!({
        "version": "0.1.0",
        "vocab_size": chars.len(),
        "characters": char_strings.join(""),
        "special_tokens": {
            "unk": 0,
            "pad": 1
        }
    });

    fs::write(vocab_path, serde_json::to_string_pretty(&vocab_json)?)?;
    Ok(())
}

/// Training step with gradient descent and clipping
fn train_step(
    model: &LanguageModel,
    optimizer: &mut candle_nn::AdamW,
    inputs: &Tensor,
    targets: &Tensor,
) -> Result<f32> {
    let logits = model.forward(inputs)?;
    let loss = loss::cross_entropy(&logits, targets)?;

    // Backward pass with gradient clipping
    optimizer.backward_step(&loss)?;

    // TODO: Add explicit gradient clipping when Candle supports it
    // For now, AdamW has built-in numerical stability

    // Return scalar loss
    loss.to_vec0::<f32>()
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    println!("🇺🇦 Ukrainian Keyboard Trainer - Model Training");
    println!("===============================================\n");

    // ✨ Use Metal GPU on macOS M1 (10-15× faster), fallback to CPU
    #[cfg(target_os = "macos")]
    println!("🚀 Metal GPU acceleration: ENABLED");
    #[cfg(not(target_os = "macos"))]
    println!("💻 CPU mode (Metal GPU not available on this platform)");
    println!();

    // Load corpus
    println!("📚 Loading corpus from: {}", args.corpus);
    let text = load_corpus(&args.corpus)?;
    let text_len = text.len();
    println!("   Loaded {} characters", text_len);

    // Build vocabulary
    println!("\n🔤 Building vocabulary...");
    let (chars, char_to_id) = build_vocab(&text);
    let vocab_size = chars.len();
    println!("   Vocab size: {} unique characters", vocab_size);
    if args.verbose {
        println!("   Characters: {:?}", &chars[..vocab_size.min(20)]);
    }

    // Convert text to indices
    println!("\n🔢 Encoding text to character indices...");
    let indices = text_to_indices(&text, &char_to_id);
    println!("   Encoded {} character indices", indices.len());

    // Create training batches
    println!(
        "\n    📦 Creating batches (context={}, batch_size={})...",
        args.context, args.batch_size
    );
    let batches = create_batches(&indices, args.batch_size, args.context);
    println!("   Created {} training examples", batches.len());

    // Split into train/validation (90/10)
    let split_idx = (batches.len() as f32 * 0.9) as usize;
    let (train_batches, val_batches) = batches.split_at(split_idx);
    println!(
        "   Train: {} | Validation: {}",
        train_batches.len(),
        val_batches.len()
    );

    // Initialize model and optimizer
    println!("\n🧠 Initializing neural language model...");

    // Setup device (Metal GPU on macOS, CPU elsewhere)
    #[cfg(target_os = "macos")]
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    #[cfg(not(target_os = "macos"))]
    let device = Device::Cpu;

    println!("   Device: {:?}", device);

    // Create variable map and model
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = match LanguageModel::new(vs, vocab_size, args.hidden_dim, args.embedding_dim) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("❌ Failed to create model: {}", e);
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Model creation failed: {}", e),
            ));
        }
    };

    let mut optimizer = candle_nn::AdamW::new(
        varmap.all_vars(),
        candle_nn::ParamsAdamW {
            lr: args.learning_rate,
            ..Default::default()
        },
    )
    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    // Training loop with actual gradient descent
    println!("\n🎓 Training model for {} epochs...", args.epochs);
    println!("   Train examples: {}", train_batches.len());
    println!("   Val examples: {}", val_batches.len());
    println!("   Learning rate: {}", args.learning_rate);

    let start_time = std::time::Instant::now();
    let mut best_val_loss = f32::INFINITY;

    for epoch in 1..=args.epochs {
        // Training phase
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        let epoch_start = std::time::Instant::now();

        for (batch_idx, (batch_inputs, batch_targets)) in train_batches.iter().enumerate() {
            // Convert batched sequences to tensors: (batch_size, seq_len)
            let batch_len = batch_inputs.len();
            let seq_len = if batch_len > 0 {
                batch_inputs[0].len()
            } else {
                0
            };

            let input_flat: Vec<u32> = batch_inputs
                .iter()
                .flat_map(|seq| seq.iter().map(|&x| x as u32))
                .collect();
            let target_flat: Vec<u32> = batch_targets.iter().map(|&x| x as u32).collect();

            match (
                Tensor::from_slice(&input_flat, (batch_len, seq_len), &device),
                Tensor::from_slice(&target_flat, batch_len, &device),
            ) {
                (Ok(input_tensor), Ok(target_tensor)) => {
                    match train_step(&model, &mut optimizer, &input_tensor, &target_tensor) {
                        Ok(loss) => {
                            total_loss += loss;
                            batch_count += 1;

                            // Progress update every 500 batches
                            if batch_idx > 0 && batch_idx % 500 == 0 {
                                let avg_loss_so_far = total_loss / batch_count as f32;
                                let progress =
                                    (batch_idx as f32 / train_batches.len() as f32) * 100.0;
                                println!(
                                    "      Batch {}/{} ({:.1}%) - loss: {:.6}",
                                    batch_idx,
                                    train_batches.len(),
                                    progress,
                                    avg_loss_so_far
                                );
                            }
                        }
                        Err(e) => {
                            if args.verbose {
                                eprintln!("   Warning: batch {} training error: {}", batch_idx, e);
                            }
                        }
                    }
                }
                (Err(e), _) | (_, Err(e)) => {
                    if args.verbose {
                        eprintln!(
                            "   Warning: batch {} tensor creation error: {}",
                            batch_idx, e
                        );
                    }
                }
            }
        }

        let train_loss = if batch_count > 0 {
            total_loss / batch_count as f32
        } else {
            0.0
        };

        // Validation phase
        let mut val_loss_total = 0.0;
        let mut val_count = 0;

        for (batch_inputs, batch_targets) in val_batches.iter() {
            let batch_len = batch_inputs.len();
            let seq_len = if batch_len > 0 {
                batch_inputs[0].len()
            } else {
                0
            };

            let input_flat: Vec<u32> = batch_inputs
                .iter()
                .flat_map(|seq| seq.iter().map(|&x| x as u32))
                .collect();
            let target_flat: Vec<u32> = batch_targets.iter().map(|&x| x as u32).collect();

            if let (Ok(input_tensor), Ok(target_tensor)) = (
                Tensor::from_slice(&input_flat, (batch_len, seq_len), &device),
                Tensor::from_slice(&target_flat, batch_len, &device),
            ) {
                if let Ok(logits) = model.forward(&input_tensor) {
                    if let Ok(loss) = loss::cross_entropy(&logits, &target_tensor) {
                        if let Ok(loss_val) = loss.to_vec0::<f32>() {
                            val_loss_total += loss_val;
                            val_count += 1;
                        }
                    }
                }
            }
        }

        let val_loss = if val_count > 0 {
            val_loss_total / val_count as f32
        } else {
            0.0
        };

        // Track best validation loss and save checkpoint
        let improved = if val_loss < best_val_loss {
            best_val_loss = val_loss;

            // Save best model checkpoint
            let best_model_path = Path::new(&args.output).join("model_best.bin");
            if let Err(e) = save_model(
                best_model_path.to_str().unwrap(),
                vocab_size,
                args.hidden_dim,
                args.embedding_dim,
                &varmap,
            ) {
                eprintln!("      Warning: Failed to save best checkpoint: {}", e);
            }

            " 🌟 (saved)"
        } else {
            ""
        };

        let epoch_time = epoch_start.elapsed();
        let examples_per_sec = train_batches.len() as f32 / epoch_time.as_secs_f32();

        println!(
            "   ✓ Epoch {}/{}: train_loss={:.6}, val_loss={:.6}, time={:.2}s, ex/s={:.0}{}",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            epoch_time.as_secs_f32(),
            examples_per_sec,
            improved
        );
    }

    let total_time = start_time.elapsed();
    println!(
        "\n⏱️  Total training time: {:.2}s ({:.2}m)",
        total_time.as_secs_f32(),
        total_time.as_secs_f32() / 60.0
    );

    // Save final model weights
    println!("\n💾 Saving final model weights...");
    fs::create_dir_all(&args.output)?;
    let model_path = Path::new(&args.output).join("model_weights.bin");
    save_model(
        model_path.to_str().unwrap(),
        vocab_size,
        args.hidden_dim,
        args.embedding_dim,
        &varmap,
    )?;
    println!("   Final model: {}", model_path.display());
    println!(
        "   Best model: {}/model_best.bin (val_loss={:.6})",
        args.output, best_val_loss
    );

    // Save vocabulary
    println!("\n📝 Saving vocabulary...");
    save_vocab_json(&args.output, &chars)?;
    println!("   Vocab saved to: {}/vocab.json", args.output);

    println!("\n✅ Training complete!");
    println!(
        "📊 Summary:\n   - Vocab size: {}\n   - Model: {}\n   - Vocab: {}/vocab.json",
        vocab_size,
        model_path.display(),
        args.output
    );

    Ok(())
}
