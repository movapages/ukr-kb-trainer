//! Tokenizer: Character-level vocabulary for Ukrainian
//!
//! Handles:
//! - Character to token ID mapping
//! - Token ID to character reverse mapping
//! - Normalization of special characters (apostrophes, etc.)

use rustc_hash::FxHashMap;
use serde_json::json;
use std::fs;

/// Character-level tokenizer for Ukrainian
pub struct Vocab {
    /// Character → Token ID mapping
    char_to_id: FxHashMap<char, u32>,
    /// Token ID → Character reverse mapping
    id_to_char: FxHashMap<u32, char>,
    /// Special tokens
    #[allow(dead_code)]
    unk_token: u32,
    #[allow(dead_code)]
    pad_token: u32,
}

#[allow(dead_code)]
impl Vocab {
    /// Create a new vocabulary from scratch
    pub fn new() -> Self {
        let mut char_to_id = FxHashMap::default();
        let mut id_to_char = FxHashMap::default();

        // Reserve special tokens (0, 1)
        let unk_token = 0;
        let pad_token = 1;

        char_to_id.insert('\0', unk_token); // unknown
        char_to_id.insert(' ', pad_token); // padding
        id_to_char.insert(unk_token, '\0');
        id_to_char.insert(pad_token, ' ');

        Vocab {
            char_to_id,
            id_to_char,
            unk_token,
            pad_token,
        }
    }

    /// Load vocabulary from JSON file (or build from characters if not found)
    pub fn load(vocab_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut vocab = Vocab::new();

        // Try to load from file; if not found, build from Ukrainian alphabet
        if let Ok(content) = fs::read_to_string(vocab_path) {
            let json: serde_json::Value = serde_json::from_str(&content)?;
            if let Some(chars) = json.get("characters").and_then(|v| v.as_array()) {
                let mut token_id = 2; // Start after special tokens
                for char_val in chars {
                    if let Some(char_str) = char_val.as_str() {
                        for c in char_str.chars() {
                            vocab.char_to_id.insert(c, token_id);
                            vocab.id_to_char.insert(token_id, c);
                            token_id += 1;
                        }
                    }
                }
            }
            Ok(vocab)
        } else {
            // Build default Ukrainian alphabet
            vocab.add_ukrainian_alphabet();
            Ok(vocab)
        }
    }

    /// Add full Ukrainian alphabet
    pub fn add_ukrainian_alphabet(&mut self) {
        // Ukrainian alphabet (33 letters) - no Ё/ё (Russian)
        // Lowercase: а б в г ґ д е є ж з и і ї й к л м н о п р с т у ф х ц ч ш щ ь ю я
        // Uppercase: А Б В Г Ґ Д Е Є Ж З И І Ї Й К Л М Н О П Р С Т У Ф Х Ц Ч Ш Щ Ь Ю Я
        let ukrainian_chars = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюяАБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ0123456789-–—.,;:!?\"'`ʼ()[]{}«»\n\t";
        let mut token_id = 2; // Start after special tokens

        for c in ukrainian_chars.chars() {
            if !self.char_to_id.contains_key(&c) {
                self.char_to_id.insert(c, token_id);
                self.id_to_char.insert(token_id, c);
                token_id += 1;
            }
        }
    }

    /// Normalize apostrophes and special characters
    fn normalize(text: &str) -> String {
        text.replace('ʼ', "'").replace('`', "'")
    }

    /// Convert character to token ID
    pub fn char_to_token(&self, c: char) -> Option<u32> {
        self.char_to_id.get(&c).copied()
    }

    /// Convert token ID to character
    pub fn token_to_char(&self, token_id: u32) -> Option<char> {
        self.id_to_char.get(&token_id).copied()
    }

    /// Encode string to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let normalized = Self::normalize(text);
        let mut tokens = Vec::new();

        for c in normalized.chars() {
            let token = self.char_to_id.get(&c).copied().unwrap_or(self.unk_token);
            tokens.push(token);
        }

        Ok(tokens)
    }

    /// Decode token IDs to string
    pub fn decode(&self, tokens: &[u32]) -> Result<String, Box<dyn std::error::Error>> {
        let mut text = String::new();

        for &token_id in tokens {
            if let Some(c) = self.id_to_char.get(&token_id) {
                if *c != '\0' {
                    // Skip unknown token
                    text.push(*c);
                }
            }
        }

        Ok(text)
    }

    /// Get vocabulary size
    pub fn size(&self) -> usize {
        self.char_to_id.len()
    }

    /// Save vocabulary to JSON
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let chars: Vec<char> = self.char_to_id.keys().copied().collect();
        let data = json!({
            "version": "0.1.0",
            "vocab_size": self.char_to_id.len(),
            "characters": chars.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(""),
            "special_tokens": {
                "unk": self.unk_token,
                "pad": self.pad_token
            }
        });

        fs::write(path, serde_json::to_string_pretty(&data)?)?;
        Ok(())
    }
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let mut vocab = Vocab::new();
        vocab.add_ukrainian_alphabet();
        let text = "мама";
        let tokens = vocab.encode(text).unwrap();
        let decoded = vocab.decode(&tokens).unwrap();
        assert_eq!(text, decoded);
    }

    #[test]
    fn test_apostrophe_normalization() {
        let mut vocab = Vocab::new();
        vocab.add_ukrainian_alphabet();
        let text = "дʼ";
        let normalized = Vocab::normalize(text);
        assert_eq!(normalized, "д'");
    }
}
