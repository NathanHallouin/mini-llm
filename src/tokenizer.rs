//! Simple character-level tokenizer.
//!
//! A tokenizer converts text into numbers (tokens) that the model can process.
//! This is the first step in any LLM pipeline.
//!
//! Two main approaches exist:
//! - Character-level: each character = 1 token (simple, used here)
//! - Subword (BPE, WordPiece): word pieces = tokens (GPT, BERT)

use std::collections::HashMap;

/// Character-level tokenizer.
///
/// # Advantages
/// - Small, fixed vocabulary
/// - Can handle any text
/// - Simple to understand
///
/// # Disadvantages
/// - Longer sequences
/// - Model must learn spelling
#[derive(Clone)]
pub struct CharTokenizer {
    /// Character to index mapping
    char_to_idx: HashMap<char, usize>,
    /// Index to character mapping
    idx_to_char: HashMap<usize, char>,
    /// Vocabulary size
    vocab_size: usize,
}

impl CharTokenizer {
    /// Creates a new empty tokenizer.
    pub fn new() -> Self {
        Self {
            char_to_idx: HashMap::new(),
            idx_to_char: HashMap::new(),
            vocab_size: 0,
        }
    }

    /// Builds the vocabulary from the given text.
    ///
    /// Finds all unique characters and creates bidirectional mappings.
    pub fn fit(&mut self, text: &str) {
        // Collect all unique characters and sort them
        let mut chars: Vec<char> = text.chars().collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        chars.sort();

        // Create the mappings
        for (idx, ch) in chars.iter().enumerate() {
            self.char_to_idx.insert(*ch, idx);
            self.idx_to_char.insert(idx, *ch);
        }

        self.vocab_size = chars.len();
        println!("Vocabulary created: {} unique characters", self.vocab_size);
    }

    /// Converts text to a list of tokens (numbers).
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|ch| self.char_to_idx.get(&ch).copied())
            .collect()
    }

    /// Converts tokens back to text.
    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .filter_map(|idx| self.idx_to_char.get(idx))
            .collect()
    }

    /// Returns the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

impl Default for CharTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_roundtrip() {
        let mut tokenizer = CharTokenizer::new();
        let text = "Hello, world!";

        tokenizer.fit(text);

        let encoded = tokenizer.encode(text);
        let decoded = tokenizer.decode(&encoded);

        assert_eq!(decoded, text);
    }
}
