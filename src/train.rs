//! Training module for the mini-LLM.
//!
//! This file demonstrates how to train a language model:
//! 1. Load and prepare the data
//! 2. Create batches
//! 3. Training loop with loss and optimizer
//!
//! The task: predict the next character (next token prediction).

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{loss, Optimizer, VarBuilder, VarMap};
use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;

use crate::model::{Config, MiniLLM};
use crate::tokenizer::CharTokenizer;

/// Training configuration
pub struct TrainConfig {
    pub seq_len: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub num_epochs: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            seq_len: 128,
            batch_size: 32,
            learning_rate: 3e-4,
            num_epochs: 10,
        }
    }
}

/// Dataset for training.
///
/// Stores tokens and allows creating batches.
pub struct TextDataset {
    tokens: Vec<usize>,
    seq_len: usize,
}

impl TextDataset {
    pub fn new(tokens: Vec<usize>, seq_len: usize) -> Self {
        println!(
            "Dataset created: {} tokens, {} possible examples",
            tokens.len(),
            tokens.len().saturating_sub(seq_len + 1)
        );
        Self { tokens, seq_len }
    }

    /// Returns the number of examples in the dataset.
    pub fn len(&self) -> usize {
        self.tokens.len().saturating_sub(self.seq_len + 1)
    }

    /// Creates a batch of data.
    ///
    /// For each example:
    /// - Input: tokens[i..i+seq_len]
    /// - Target: tokens[i+1..i+seq_len+1]
    pub fn get_batch(
        &self,
        indices: &[usize],
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let batch_size = indices.len();

        let mut inputs = Vec::with_capacity(batch_size * self.seq_len);
        let mut targets = Vec::with_capacity(batch_size * self.seq_len);

        for &idx in indices {
            let chunk = &self.tokens[idx..idx + self.seq_len + 1];

            // Input: all but the last
            for &t in &chunk[..self.seq_len] {
                inputs.push(t as u32);
            }

            // Target: all but the first (shifted by 1)
            for &t in &chunk[1..] {
                targets.push(t as u32);
            }
        }

        let x = Tensor::from_vec(inputs, (batch_size, self.seq_len), device)?;
        let y = Tensor::from_vec(targets, (batch_size, self.seq_len), device)?;

        Ok((x, y))
    }
}

/// Trains the model.
pub fn train(
    data_path: &str,
    model_config: Config,
    train_config: TrainConfig,
    device: &Device,
) -> Result<(MiniLLM, CharTokenizer, VarMap)> {
    // 1. Load data
    println!("\nLoading {}...", data_path);
    let text = std::fs::read_to_string(data_path)?;
    println!("Text loaded: {} characters", text.len());

    // 2. Tokenization
    println!("\nCreating tokenizer...");
    let mut tokenizer = CharTokenizer::new();
    tokenizer.fit(&text);

    let tokens = tokenizer.encode(&text);

    // 3. Create dataset
    println!("\nCreating dataset...");
    let dataset = TextDataset::new(tokens, train_config.seq_len);

    if dataset.len() == 0 {
        anyhow::bail!("Dataset too small! Add more text.");
    }

    // 4. Create model
    println!("\nCreating model...");
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

    let config = Config {
        vocab_size: tokenizer.vocab_size(),
        ..model_config
    };

    let model = MiniLLM::new(config, vb)?;

    // 5. Optimizer (AdamW)
    let mut optimizer = candle_nn::AdamW::new(
        varmap.all_vars(),
        candle_nn::ParamsAdamW {
            lr: train_config.learning_rate,
            weight_decay: 0.01,
            ..Default::default()
        },
    )?;

    // 6. Indices for shuffling
    let mut indices: Vec<usize> = (0..dataset.len()).collect();
    let num_batches = (dataset.len() + train_config.batch_size - 1) / train_config.batch_size;

    // 7. Training loop
    println!(
        "\nStarting training ({} epochs, {} batches/epoch)...\n",
        train_config.num_epochs, num_batches
    );

    for epoch in 0..train_config.num_epochs {
        // Shuffle indices
        indices.shuffle(&mut rand::thread_rng());

        let mut total_loss = 0.0;
        let mut num_batches_done = 0;

        // Progress bar
        let pb = ProgressBar::new(num_batches as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} loss: {msg}")
                .unwrap()
                .progress_chars("##-"),
        );

        for batch_start in (0..dataset.len()).step_by(train_config.batch_size) {
            let batch_end = (batch_start + train_config.batch_size).min(dataset.len());
            let batch_indices = &indices[batch_start..batch_end];

            // Get batch
            let (x, y) = dataset.get_batch(batch_indices, device)?;

            // Forward pass
            let logits = model.forward(&x)?;

            // Reshape for loss: [batch * seq, vocab] and [batch * seq]
            let (batch_size, seq_len, vocab_size) = logits.dims3()?;
            let logits_flat = logits.reshape((batch_size * seq_len, vocab_size))?;
            let targets_flat = y.reshape((batch_size * seq_len,))?;

            // Cross-entropy loss
            let loss = loss::cross_entropy(&logits_flat, &targets_flat)?;

            // Backward pass
            optimizer.backward_step(&loss)?;

            let loss_val: f32 = loss.to_scalar()?;
            total_loss += loss_val;
            num_batches_done += 1;

            pb.set_message(format!("{:.4}", loss_val));
            pb.inc(1);
        }

        pb.finish();

        let avg_loss = total_loss / num_batches_done as f32;
        println!("Epoch {}/{} - Average loss: {:.4}", epoch + 1, train_config.num_epochs, avg_loss);

        // Generate a sample
        println!("Sample generation:");
        let sample = generate_sample(&model, &tokenizer, device, 100)?;
        if sample.len() > 80 {
            println!("  '{}'...\n", &sample[..80]);
        } else {
            println!("  '{}'\n", sample);
        }
    }

    Ok((model, tokenizer, varmap))
}

/// Generates a text sample.
pub fn generate_sample(
    model: &MiniLLM,
    tokenizer: &CharTokenizer,
    device: &Device,
    max_tokens: usize,
) -> Result<String> {
    // Start with the first token in vocabulary
    let start = Tensor::from_vec(vec![0u32], (1, 1), device)?;

    let generated = model.generate(&start, max_tokens, 0.8)?;
    let tokens: Vec<u32> = generated.squeeze(0)?.to_vec1()?;
    let tokens: Vec<usize> = tokens.into_iter().map(|t| t as usize).collect();

    Ok(tokenizer.decode(&tokens))
}

/// Saves the model weights.
pub fn save_model(varmap: &VarMap, path: &str) -> Result<()> {
    varmap.save(path)?;
    println!("Model saved: {}", path);
    Ok(())
}

/// Loads model weights.
pub fn load_model(
    path: &str,
    config: Config,
    device: &Device,
) -> Result<(MiniLLM, VarMap)> {
    let mut varmap = VarMap::new();
    varmap.load(path)?;

    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = MiniLLM::new(config, vb)?;

    println!("Model loaded: {}", path);
    Ok((model, varmap))
}
