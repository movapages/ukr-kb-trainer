//! ScoreFusion: Multi-factor word scoring
//!
//! Combines:
//! - LM probability (0.4)
//! - Word frequency (0.3)
//! - Word length (0.2)
//! - Finger zone rules (0.1)

use rustc_hash::FxHashMap;
use std::fs;

/// Weights for ScoreFusion components
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct ScoringWeights {
    pub lm_weight: f32,
    pub frequency_weight: f32,
    pub length_weight: f32,
    pub finger_weight: f32,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        ScoringWeights {
            lm_weight: 0.4,
            frequency_weight: 0.3,
            length_weight: 0.2,
            finger_weight: 0.1,
        }
    }
}

/// Scoring engine that combines multiple factors
#[allow(dead_code)]
pub struct ScoreFusion {
    weights: ScoringWeights,
    frequencies: FxHashMap<String, f32>,
    min_length: usize,
    max_length: usize,
}

impl ScoreFusion {
    /// Create new scoring engine with default weights
    #[allow(dead_code)]
    pub fn new() -> Self {
        ScoreFusion {
            weights: ScoringWeights::default(),
            frequencies: FxHashMap::default(),
            min_length: 1,
            max_length: 20,
        }
    }

    /// Create with custom weights
    #[allow(dead_code)]
    pub fn with_weights(weights: ScoringWeights) -> Self {
        ScoreFusion {
            weights,
            frequencies: FxHashMap::default(),
            min_length: 1,
            max_length: 20,
        }
    }

    /// Load word frequencies from JSON file
    #[allow(dead_code)]
    pub fn load_frequencies(
        &mut self,
        frequencies_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let content = fs::read_to_string(frequencies_path)?;
        let json: serde_json::Value = serde_json::from_str(&content)?;

        if let Some(freqs) = json.get("frequencies").and_then(|v| v.as_object()) {
            for (word, freq_val) in freqs {
                if let Some(freq) = freq_val.as_f64() {
                    self.frequencies.insert(word.clone(), freq as f32);
                }
            }
        }

        Ok(())
    }

    /// Normalize LM logits to probabilities using softmax
    #[allow(dead_code)]
    fn softmax(logits: &[f32]) -> Vec<f32> {
        if logits.is_empty() {
            return vec![];
        }

        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();

        exps.iter().map(|&x| x / sum).collect()
    }

    /// Score a single word based on multiple factors
    #[allow(dead_code)]
    pub fn score_word(&self, word: &str, lm_score: f32, finger_zones: Option<usize>) -> f32 {
        let mut score = 0.0;

        // 1. LM score (0.4 weight)
        let lm_normalized = lm_score.max(0.0).min(1.0);
        score += self.weights.lm_weight * lm_normalized;

        // 2. Frequency score (0.3 weight)
        let freq = self
            .frequencies
            .get(word)
            .copied()
            .unwrap_or(1e-6)
            .max(1e-8);
        let log_freq = freq.ln().max(-10.0); // Prevent extreme values
        let freq_normalized = (log_freq + 10.0) / 10.0; // Normalize to [0, 1]
        score += self.weights.frequency_weight * freq_normalized.max(0.0).min(1.0);

        // 3. Length score (0.2 weight) - prefer moderate lengths
        let length = word.len() as f32;
        let ideal_length = (self.min_length + self.max_length) as f32 / 2.0;
        let length_distance = (length - ideal_length).abs();
        let max_distance = (self.max_length as f32 - self.min_length as f32) / 2.0;
        let length_normalized = 1.0 - (length_distance / max_distance).min(1.0);
        score += self.weights.length_weight * length_normalized;

        // 4. Finger zone score (0.1 weight)
        let finger_score = if let Some(zones) = finger_zones {
            (zones as f32 / 8.0).min(1.0)
        } else {
            0.5 // Neutral if not specified
        };
        score += self.weights.finger_weight * finger_score;

        score
    }

    /// Score candidate words based on multiple factors
    #[allow(dead_code)]
    pub fn score(
        &self,
        candidates: &[String],
        lm_logits: &[f32],
        finger_zones: Option<Vec<usize>>,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        if candidates.len() != lm_logits.len() {
            return Err("Candidates and logits must have same length".into());
        }

        let scores: Vec<f32> = candidates
            .iter()
            .zip(lm_logits.iter())
            .enumerate()
            .map(|(i, (word, lm_logit))| {
                let finger_score = finger_zones.as_ref().and_then(|z| z.get(i).copied());
                self.score_word(word, *lm_logit, finger_score)
            })
            .collect();

        Ok(scores)
    }

    /// Rank candidates by score    
    #[allow(dead_code)]
    pub fn rank(
        &self,
        candidates: &[String],
        scores: &[f32],
        top_k: usize,
    ) -> Result<Vec<(String, f32)>, Box<dyn std::error::Error>> {
        if candidates.len() != scores.len() {
            return Err("Candidates and scores must have same length".into());
        }

        let mut ranked: Vec<(String, f32)> = candidates
            .iter()
            .zip(scores.iter())
            .map(|(word, score)| (word.clone(), *score))
            .collect();

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(top_k);

        Ok(ranked)
    }
}

impl Default for ScoreFusion {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = ScoreFusion::softmax(&logits);
        assert_eq!(probs.len(), 3);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_score_word() {
        let scorer = ScoreFusion::new();
        let score = scorer.score_word("мама", 0.8, Some(2));
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }
}
