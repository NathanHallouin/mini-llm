//! Simplified Transformer architecture for a mini-LLM.
//!
//! This file contains the essential components of a Transformer:
//! 1. Embedding (text -> vectors)
//! 2. Positional Encoding (adds position information)
//! 3. Self-Attention (the heart of the Transformer)
//! 4. Feed-Forward Network
//! 5. The complete model
//!
//! References:
//! - "Attention Is All You Need" (original paper)
//! - "The Illustrated Transformer" (Jay Alammar's blog)

use candle_core::{Device, Module, Result, Tensor, D};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, VarBuilder};

/// Model configuration
#[derive(Clone, Debug)]
pub struct Config {
    pub vocab_size: usize,
    pub d_model: usize,      // Embedding dimension
    pub num_heads: usize,    // Number of attention heads
    pub num_layers: usize,   // Number of Transformer blocks
    pub d_ff: usize,         // Feed-forward hidden dimension
    pub max_seq_len: usize,  // Maximum sequence length
    pub dropout: f64,        // Dropout rate (unused in inference)
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 100,
            d_model: 128,
            num_heads: 4,
            num_layers: 4,
            d_ff: 512,
            max_seq_len: 256,
            dropout: 0.1,
        }
    }
}

/// Multi-Head Self-Attention - The core of the Transformer.
///
/// Attention allows the model to "look at" other parts of the sequence
/// when processing each token. "Multi-head" = multiple perspectives.
///
/// Key concepts:
/// - Query (Q): "What am I looking for?"
/// - Key (K): "What do I have to offer?"
/// - Value (V): "What information do I carry?"
///
/// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
pub struct MultiHeadAttention {
    d_model: usize,
    num_heads: usize,
    d_k: usize,
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        assert!(d_model % num_heads == 0, "d_model must be divisible by num_heads");

        let d_k = d_model / num_heads;

        // Linear projections for Q, K, V
        let w_q = linear(d_model, d_model, vb.pp("w_q"))?;
        let w_k = linear(d_model, d_model, vb.pp("w_k"))?;
        let w_v = linear(d_model, d_model, vb.pp("w_v"))?;
        let w_o = linear(d_model, d_model, vb.pp("w_o"))?;

        Ok(Self { d_model, num_heads, d_k, w_q, w_k, w_v, w_o })
    }

    /// Forward pass of the attention mechanism.
    ///
    /// # Arguments
    /// * `x` - Tensor of shape [batch_size, seq_len, d_model]
    /// * `mask` - Optional causal mask
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // 1. Linear projections
        let q = self.w_q.forward(x)?;
        let k = self.w_k.forward(x)?;
        let v = self.w_v.forward(x)?;

        // 2. Reshape for multi-head: [batch, seq, num_heads, d_k] -> [batch, num_heads, seq, d_k]
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.d_k))?.transpose(1, 2)?;
        let k = k.reshape((batch_size, seq_len, self.num_heads, self.d_k))?.transpose(1, 2)?;
        let v = v.reshape((batch_size, seq_len, self.num_heads, self.d_k))?.transpose(1, 2)?;

        // 3. Compute attention: softmax(QK^T / sqrt(d_k)) * V
        let scale = (self.d_k as f64).sqrt();
        let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let scores = (scores / scale)?;

        // 4. Apply causal mask (for autoregressive generation)
        let scores = match mask {
            Some(m) => scores.broadcast_add(m)?,
            None => scores,
        };

        let attention_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;

        // 5. Multiply by V
        let context = attention_weights.matmul(&v)?;

        // 6. Reshape and final projection: [batch, num_heads, seq, d_k] -> [batch, seq, d_model]
        let context = context.transpose(1, 2)?.reshape((batch_size, seq_len, self.d_model))?;

        self.w_o.forward(&context)
    }
}

/// Feed-Forward Network - Position-wise processing.
///
/// After attention (which mixes information across positions), each position
/// is processed independently by this small network.
///
/// Structure: Linear -> ReLU -> Linear
pub struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        let linear1 = linear(d_model, d_ff, vb.pp("linear1"))?;
        let linear2 = linear(d_ff, d_model, vb.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = x.relu()?;
        self.linear2.forward(&x)
    }
}

/// A Transformer block = Attention + Feed-Forward + Residual connections.
///
/// Structure:
/// x -> LayerNorm -> Attention -> + -> LayerNorm -> FFN -> +
/// |___________________________|   |_____________________|
///       (residual connection)        (residual connection)
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl TransformerBlock {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let attention = MultiHeadAttention::new(config.d_model, config.num_heads, vb.pp("attention"))?;
        let feed_forward = FeedForward::new(config.d_model, config.d_ff, vb.pp("ff"))?;
        let norm1 = layer_norm(config.d_model, 1e-5, vb.pp("norm1"))?;
        let norm2 = layer_norm(config.d_model, 1e-5, vb.pp("norm2"))?;

        Ok(Self { attention, feed_forward, norm1, norm2 })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // Pre-norm architecture (more stable for training)
        let normed = self.norm1.forward(x)?;
        let attn_output = self.attention.forward(&normed, mask)?;
        let x = (x + attn_output)?; // Residual connection

        let normed = self.norm2.forward(&x)?;
        let ff_output = self.feed_forward.forward(&normed)?;

        x + ff_output // Residual connection
    }
}

/// The complete model - A mini Language Model.
///
/// Architecture:
/// 1. Token Embedding: converts tokens to vectors
/// 2. Positional Embedding: adds position information
/// 3. N stacked Transformer blocks
/// 4. Final Layer Norm
/// 5. Projection to vocabulary (to predict the next token)
pub struct MiniLLM {
    token_embedding: Embedding,
    pos_embedding: Embedding,  // Learned positional embedding (simpler than sinusoidal)
    layers: Vec<TransformerBlock>,
    norm: LayerNorm,
    output_projection: Linear,
    config: Config,
}

impl MiniLLM {
    pub fn new(config: Config, vb: VarBuilder) -> Result<Self> {
        // Token embedding
        let token_embedding = embedding(config.vocab_size, config.d_model, vb.pp("token_emb"))?;

        // Positional embedding (learned version, simpler than sinusoidal)
        let pos_embedding = embedding(config.max_seq_len, config.d_model, vb.pp("pos_emb"))?;

        // Transformer blocks
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            layers.push(TransformerBlock::new(&config, vb.pp(format!("layer_{}", i)))?);
        }

        // Final normalization
        let norm = layer_norm(config.d_model, 1e-5, vb.pp("norm"))?;

        // Output projection to vocabulary
        let output_projection = linear(config.d_model, config.vocab_size, vb.pp("output"))?;

        // Count parameters
        let num_params = count_params(&vb);
        println!("Model created with approximately {} parameters", num_params);

        Ok(Self {
            token_embedding,
            pos_embedding,
            layers,
            norm,
            output_projection,
            config,
        })
    }

    /// Creates a causal (triangular) attention mask.
    ///
    /// Prevents each position from seeing future positions.
    /// Forbidden positions have a value of -inf.
    fn create_causal_mask(&self, seq_len: usize, device: &Device) -> Result<Tensor> {
        // Manually create a lower triangular matrix
        // 0 = can see, -inf = cannot see
        let mut mask_data = vec![0.0f32; seq_len * seq_len];

        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    // Position j is in the future of position i -> mask it
                    mask_data[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
        }

        let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), device)?;

        // Add dimensions for broadcasting: [1, 1, seq, seq]
        mask.unsqueeze(0)?.unsqueeze(0)
    }

    /// Forward pass of the model.
    ///
    /// # Arguments
    /// * `x` - Token tensor [batch_size, seq_len]
    ///
    /// # Returns
    /// Logits of shape [batch_size, seq_len, vocab_size]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_batch_size, seq_len) = x.dims2()?;
        let device = x.device();

        // 1. Token embedding
        let mut hidden = self.token_embedding.forward(x)?;

        // 2. Positional embedding
        let positions = Tensor::arange(0u32, seq_len as u32, device)?;
        let pos_emb = self.pos_embedding.forward(&positions)?;
        hidden = hidden.broadcast_add(&pos_emb)?;

        // 3. Causal mask
        let mask = self.create_causal_mask(seq_len, device)?;

        // 4. Pass through Transformer blocks
        for layer in &self.layers {
            hidden = layer.forward(&hidden, Some(&mask))?;
        }

        // 5. Normalize and project
        let hidden = self.norm.forward(&hidden)?;
        self.output_projection.forward(&hidden)
    }

    /// Generates text autoregressively.
    ///
    /// At each step:
    /// 1. Pass the current sequence through the model
    /// 2. Get logits for the last position
    /// 3. Sample the next token
    /// 4. Append to sequence and repeat
    pub fn generate(
        &self,
        start_tokens: &Tensor,
        max_new_tokens: usize,
        temperature: f64,
    ) -> Result<Tensor> {
        let device = start_tokens.device();
        let mut tokens = start_tokens.clone();

        for _ in 0..max_new_tokens {
            // Truncate if too long
            let seq_len = tokens.dims()[1];
            let x = if seq_len > self.config.max_seq_len {
                tokens.narrow(1, seq_len - self.config.max_seq_len, self.config.max_seq_len)?
            } else {
                tokens.clone()
            };

            // Forward pass
            let logits = self.forward(&x)?;

            // Take the last position: [batch, vocab]
            let last_logits = logits.narrow(1, logits.dims()[1] - 1, 1)?.squeeze(1)?;

            // Apply temperature
            let scaled_logits = (last_logits / temperature)?;

            // Softmax to get probabilities
            let probs = candle_nn::ops::softmax(&scaled_logits, D::Minus1)?;

            // Sample next token
            let next_token = sample_from_probs(&probs, device)?;

            // Append to sequence
            tokens = Tensor::cat(&[&tokens, &next_token.unsqueeze(1)?], 1)?;
        }

        Ok(tokens)
    }

    pub fn config(&self) -> &Config {
        &self.config
    }
}

/// Samples a token from the probability distribution.
fn sample_from_probs(probs: &Tensor, device: &Device) -> Result<Tensor> {
    let probs_vec: Vec<f32> = probs.squeeze(0)?.to_vec1()?;

    // Simple weighted sampling
    let mut rng = rand::thread_rng();
    let dist = rand::distributions::WeightedIndex::new(&probs_vec)
        .map_err(|e| candle_core::Error::Msg(format!("Sampling error: {}", e)))?;

    use rand::distributions::Distribution;
    let idx = dist.sample(&mut rng) as u32;

    Tensor::from_vec(vec![idx], (1,), device)
}

/// Estimates the number of parameters (approximate).
fn count_params(_vb: &VarBuilder) -> String {
    // Note: Candle doesn't directly provide this info
    // This is an estimate based on typical config
    "~500K".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_model_forward() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = Config {
            vocab_size: 100,
            d_model: 64,
            num_heads: 4,
            num_layers: 2,
            d_ff: 256,
            max_seq_len: 128,
            dropout: 0.0,
        };

        let model = MiniLLM::new(config, vb)?;

        // Test forward pass
        let input = Tensor::zeros((2, 32), DType::U32, &device)?;
        let output = model.forward(&input)?;

        assert_eq!(output.dims(), &[2, 32, 100]);
        println!("Forward test passed!");

        Ok(())
    }
}
