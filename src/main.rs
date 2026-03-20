//! Mini-LLM: A small educational language model in Rust.
//!
//! This project demonstrates how to build an LLM from scratch with Candle.
//!
//! Usage:
//! ```bash
//! # Train
//! cargo run --release -- train --data data/input.txt
//!
//! # Generate
//! cargo run --release -- generate --prompt "Once upon"
//! ```

mod model;
mod tokenizer;
mod train;

use anyhow::Result;
use candle_core::Device;
use std::io::{self, Write};

use crate::model::Config;
use crate::tokenizer::CharTokenizer;
use crate::train::TrainConfig;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Determine device
    let device = Device::Cpu; // Use Device::cuda_if_available(0)? for GPU

    println!("╔════════════════════════════════════════╗");
    println!("║      Mini-LLM - Educational Model      ║");
    println!("╠════════════════════════════════════════╣");
    println!("║  Device: {:30} ║", format!("{:?}", device));
    println!("╚════════════════════════════════════════╝");

    if args.len() < 2 {
        print_help();
        return Ok(());
    }

    match args[1].as_str() {
        "train" => {
            let data_path = get_arg(&args, "--data").unwrap_or("data/input.txt".to_string());
            let epochs: usize = get_arg(&args, "--epochs")
                .and_then(|s| s.parse().ok())
                .unwrap_or(10);
            let d_model: usize = get_arg(&args, "--d-model")
                .and_then(|s| s.parse().ok())
                .unwrap_or(128);
            let num_layers: usize = get_arg(&args, "--num-layers")
                .and_then(|s| s.parse().ok())
                .unwrap_or(4);

            let model_config = Config {
                d_model,
                num_heads: 4,
                num_layers,
                d_ff: d_model * 4,
                max_seq_len: 128,
                dropout: 0.1,
                ..Default::default()
            };

            let train_config = TrainConfig {
                num_epochs: epochs,
                ..Default::default()
            };

            let (_model, _tokenizer, varmap) = train::train(
                &data_path,
                model_config,
                train_config,
                &device,
            )?;

            // Save
            std::fs::create_dir_all("checkpoints")?;
            train::save_model(&varmap, "checkpoints/model.safetensors")?;
        }

        "generate" => {
            let prompt = get_arg(&args, "--prompt").unwrap_or_default();
            let max_tokens: usize = get_arg(&args, "--max-tokens")
                .and_then(|s| s.parse().ok())
                .unwrap_or(200);
            let temperature: f64 = get_arg(&args, "--temperature")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.8);
            let data_path = get_arg(&args, "--data").unwrap_or("data/input.txt".to_string());

            // Rebuild tokenizer from data
            let text = std::fs::read_to_string(&data_path)?;
            let mut tokenizer = CharTokenizer::new();
            tokenizer.fit(&text);

            let config = Config {
                vocab_size: tokenizer.vocab_size(),
                ..Default::default()
            };

            let (model, _varmap) = train::load_model(
                "checkpoints/model.safetensors",
                config,
                &device,
            )?;

            if prompt.is_empty() {
                // Interactive mode
                interactive_mode(&model, &tokenizer, &device, max_tokens, temperature)?;
            } else {
                // Single generation
                let output = generate_text(&model, &tokenizer, &device, &prompt, max_tokens, temperature)?;
                println!("\n{}", output);
            }
        }

        "help" | "--help" | "-h" => {
            print_help();
        }

        _ => {
            println!("Unknown command: {}", args[1]);
            print_help();
        }
    }

    Ok(())
}

fn generate_text(
    model: &model::MiniLLM,
    tokenizer: &CharTokenizer,
    device: &Device,
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
) -> Result<String> {
    use candle_core::Tensor;

    let tokens = tokenizer.encode(prompt);
    if tokens.is_empty() {
        anyhow::bail!("Empty prompt or unrecognized characters");
    }

    let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    let input = Tensor::from_vec(tokens_u32, (1, tokens.len()), device)?;

    let generated = model.generate(&input, max_tokens, temperature)?;
    let output_tokens: Vec<u32> = generated.squeeze(0)?.to_vec1()?;
    let output_tokens: Vec<usize> = output_tokens.into_iter().map(|t| t as usize).collect();

    Ok(tokenizer.decode(&output_tokens))
}

fn interactive_mode(
    model: &model::MiniLLM,
    tokenizer: &CharTokenizer,
    device: &Device,
    max_tokens: usize,
    mut temperature: f64,
) -> Result<()> {
    println!("\n════════════════════════════════════════");
    println!("Interactive mode - Enter your prompt");
    println!("Commands: 'quit', 'temp X'");
    println!("════════════════════════════════════════\n");

    loop {
        print!("Prompt> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "quit" {
            println!("Goodbye!");
            break;
        }

        if input.starts_with("temp ") {
            if let Ok(t) = input[5..].parse::<f64>() {
                temperature = t;
                println!("Temperature: {}", temperature);
            }
            continue;
        }

        match generate_text(model, tokenizer, device, input, max_tokens, temperature) {
            Ok(output) => {
                println!("\n────────────────────────────────────────");
                println!("{}", output);
                println!("────────────────────────────────────────\n");
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
    }

    Ok(())
}

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn print_help() {
    println!(
        r#"
Usage: mini-llm <COMMAND> [OPTIONS]

Commands:
  train      Train the model
  generate   Generate text

Options for 'train':
  --data <PATH>       Training text file (default: data/input.txt)
  --epochs <N>        Number of epochs (default: 10)
  --d-model <N>       Model dimension (default: 128)
  --num-layers <N>    Number of layers (default: 4)

Options for 'generate':
  --prompt <TEXT>     Starting prompt (interactive mode if absent)
  --max-tokens <N>    Max tokens to generate (default: 200)
  --temperature <F>   Temperature 0.1-2.0 (default: 0.8)
  --data <PATH>       File to rebuild tokenizer from

Examples:
  cargo run --release -- train --data data/input.txt --epochs 20
  cargo run --release -- generate --prompt "Once upon a time"
"#
    );
}
