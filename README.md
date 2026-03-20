<p align="center">
  <h1 align="center">Mini-LLM</h1>
  <p align="center">
    <strong>A minimal Transformer-based language model built from scratch in Rust</strong>
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#getting-started">Getting Started</a> •
    <a href="#usage">Usage</a> •
    <a href="#learning">Learning</a>
  </p>
</p>

---

An educational implementation of a GPT-style language model using **Rust** and [Candle](https://github.com/huggingface/candle) (Hugging Face's ML framework). This project prioritizes **clarity over performance**, making it ideal for understanding how modern LLMs work under the hood.

## Features

- **Pure Rust implementation** — No Python dependencies, single binary
- **Complete Transformer architecture** — Multi-head attention, feed-forward networks, layer normalization
- **Character-level tokenization** — Simple and easy to understand
- **Autoregressive generation** — Temperature-controlled text sampling
- **Well-documented code** — Every component explained with comments

## Architecture

```
Input Text
    │
    ▼
┌─────────────────┐
│   Tokenizer     │  "Hello" → [7, 4, 11, 11, 14]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Token Embedding │  [7, 4, ...] → [[0.1, 0.3, ...], ...]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  + Positional   │  Add position information
│    Embedding    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Transformer    │  ×N layers
│     Block       │
│  ┌───────────┐  │
│  │ Attention │  │  Self-attention mechanism
│  └───────────┘  │
│  ┌───────────┐  │
│  │    FFN    │  │  Feed-forward network
│  └───────────┘  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Output Head   │  Project to vocabulary
└────────┬────────┘
         │
         ▼
    Next Token
```

## Getting Started

### Prerequisites

- Rust 1.70+ ([install](https://rustup.rs/))

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/mini-llm.git
cd mini-llm
cargo build --release
```

### GPU Support (Optional)

Edit `Cargo.toml` to enable CUDA or Metal:

```toml
[features]
cuda = ["candle-core/cuda"]   # NVIDIA GPUs
metal = ["candle-core/metal"] # Apple Silicon
```

Then build with: `cargo build --release --features cuda`

## Usage

### Training

```bash
# Train with default settings (~500K parameters)
cargo run --release -- train

# Custom configuration
cargo run --release -- train \
    --data data/input.txt \
    --epochs 20 \
    --d-model 256 \
    --num-layers 6
```

### Text Generation

```bash
# Interactive mode
cargo run --release -- generate

# Single prompt
cargo run --release -- generate --prompt "Once upon a time"

# Adjust creativity (higher = more random)
cargo run --release -- generate --temperature 1.2
```

### CLI Reference

```
Usage: mini-llm <COMMAND> [OPTIONS]

Commands:
  train      Train the model
  generate   Generate text

Training options:
  --data <PATH>       Training text file (default: data/input.txt)
  --epochs <N>        Number of epochs (default: 10)
  --d-model <N>       Model dimension (default: 128)
  --num-layers <N>    Number of layers (default: 4)

Generation options:
  --prompt <TEXT>     Starting prompt
  --max-tokens <N>    Max tokens to generate (default: 200)
  --temperature <F>   Sampling temperature (default: 0.8)
```

## Learning

### Project Structure

```
mini-llm/
├── src/
│   ├── main.rs         # CLI entry point
│   ├── tokenizer.rs    # Character-level tokenization
│   ├── model.rs        # Transformer architecture
│   └── train.rs        # Training loop
├── data/
│   └── input.txt       # Sample training data
└── checkpoints/        # Saved models
```

### Key Concepts Explained

| File | Concept | Description |
|------|---------|-------------|
| `tokenizer.rs` | Tokenization | Converting text to numbers the model can process |
| `model.rs` | Self-Attention | How tokens "look at" other tokens to understand context |
| `model.rs` | Multi-Head Attention | Running multiple attention patterns in parallel |
| `model.rs` | Residual Connections | Skip connections that help gradients flow |
| `train.rs` | Cross-Entropy Loss | Measuring how wrong predictions are |
| `train.rs` | Backpropagation | Adjusting weights based on errors |

### Recommended Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — The original Transformer paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Visual explanations
- [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Karpathy's video tutorial
- [Candle Documentation](https://github.com/huggingface/candle) — The ML framework used here

### Model Comparison

| | Mini-LLM | GPT-2 Small | GPT-3 |
|--|---------|-------------|-------|
| Parameters | ~500K | 124M | 175B |
| Layers | 4 | 12 | 96 |
| Embedding dim | 128 | 768 | 12288 |
| Attention heads | 4 | 12 | 96 |
| Training data | ~5KB | 40GB | 570GB |

## Contributing

Contributions are welcome! This is an educational project, so clarity is more important than optimization.

## License

MIT License — See [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Built for learning. Inspired by <a href="https://github.com/karpathy/nanoGPT">nanoGPT</a>.</i>
</p>
