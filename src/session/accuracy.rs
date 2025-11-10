//! Accuracy tracking with exponential moving average
//!
//! Features:
//! - Per-finger accuracy tracking
//! - EMA for trend detection
//! - Weak finger identification

/// Tracks accuracy metrics with EMA for a session
#[derive(Clone, Debug)]
pub struct AccuracyTracker {
    /// Per-finger accuracy: [L1, L2, L3, L4, R1, R2, R3, R4]
    per_finger_accuracy: [f32; 8],
    /// Per-finger keystroke counts
    per_finger_counts: [u32; 8],
    /// Per-finger error counts
    per_finger_errors: [u32; 8],
    /// Overall EMA accuracy (alpha = 0.1)
    ema_accuracy: f32,
    /// EMA decay factor
    ema_alpha: f32,
    /// Total keystrokes typed
    total_keystrokes: u32,
    /// Total errors made
    total_errors: u32,
    /// Historical accuracy trend (last 10 words)
    accuracy_trend: Vec<f32>,
}

#[allow(dead_code)]
impl AccuracyTracker {
    /// Create new accuracy tracker
    pub fn new() -> Self {
        AccuracyTracker {
            per_finger_accuracy: [1.0; 8],
            per_finger_counts: [0; 8],
            per_finger_errors: [0; 8],
            ema_accuracy: 1.0,
            ema_alpha: 0.1,
            total_keystrokes: 0,
            total_errors: 0,
            accuracy_trend: Vec::with_capacity(10),
        }
    }

    /// Calculate accuracy for a typed word against target
    /// Returns accuracy as f32 (0.0-1.0) and a mapping of which fingers were involved
    pub fn calculate_word_accuracy(target: &str, user_input: &str) -> (f32, Vec<usize>) {
        // Compare character by character
        let target_chars: Vec<char> = target.chars().collect();
        let user_chars: Vec<char> = user_input.chars().collect();

        // Calculate basic accuracy
        let max_len = target_chars.len().max(user_chars.len());
        if max_len == 0 {
            return (1.0, vec![]);
        }

        let mut correct_count = 0;
        for i in 0..max_len {
            let target_char = target_chars.get(i);
            let user_char = user_chars.get(i);

            if target_char == user_char && target_char.is_some() {
                correct_count += 1;
            }
        }

        let accuracy = correct_count as f32 / max_len as f32;
        (accuracy, vec![])
    }

    /// Update accuracy for a typed word
    /// finger_indices: which fingers were used for this word (0-7 mapping)
    pub fn update(
        &mut self,
        target: &str,
        user_input: &str,
        finger_indices: &[usize],
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let (accuracy, _) = Self::calculate_word_accuracy(target, user_input);

        // Count errors
        let target_len = target.len();
        let errors = target_len.saturating_sub((accuracy * target_len as f32) as usize);

        // Update global stats
        self.total_keystrokes += target_len as u32;
        self.total_errors += errors as u32;

        // Update per-finger stats
        if !finger_indices.is_empty() {
            let per_finger_accuracy = accuracy / finger_indices.len() as f32;
            let per_finger_errors = (errors as f32 / finger_indices.len() as f32).ceil() as u32;

            for &finger_idx in finger_indices {
                if finger_idx < 8 {
                    let count = self.per_finger_counts[finger_idx];
                    let new_avg = (self.per_finger_accuracy[finger_idx] * count as f32
                        + per_finger_accuracy)
                        / (count as f32 + 1.0);
                    self.per_finger_accuracy[finger_idx] = new_avg;
                    self.per_finger_counts[finger_idx] += 1;
                    self.per_finger_errors[finger_idx] += per_finger_errors;
                }
            }
        }

        // Update EMA
        self.ema_accuracy = self.ema_alpha * accuracy + (1.0 - self.ema_alpha) * self.ema_accuracy;

        // Track trend
        self.accuracy_trend.push(accuracy);
        if self.accuracy_trend.len() > 10 {
            self.accuracy_trend.remove(0);
        }

        Ok(accuracy)
    }

    /// Get current EMA accuracy
    pub fn get_ema(&self) -> f32 {
        self.ema_accuracy
    }

    /// Get per-finger accuracy
    pub fn get_per_finger_accuracy(&self) -> &[f32; 8] {
        &self.per_finger_accuracy
    }

    /// Get per-finger error rate (errors / keystrokes)
    pub fn get_per_finger_error_rates(&self) -> [f32; 8] {
        let mut rates = [0.0; 8];
        for i in 0..8 {
            if self.per_finger_counts[i] > 0 {
                rates[i] = self.per_finger_errors[i] as f32 / self.per_finger_counts[i] as f32;
            }
        }
        rates
    }

    /// Identify weak fingers (< 80% accuracy)
    pub fn weak_fingers(&self) -> Vec<usize> {
        self.per_finger_accuracy
            .iter()
            .enumerate()
            .filter_map(|(idx, &acc)| {
                if self.per_finger_counts[idx] > 0 && acc < 0.80 {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get accuracy trend (recent words)
    pub fn get_trend(&self) -> &[f32] {
        &self.accuracy_trend
    }

    /// Check if accuracy is declining (recent avg lower than overall)
    pub fn is_declining(&self) -> bool {
        if self.accuracy_trend.len() < 5 {
            return false;
        }

        let recent_avg: f32 =
            self.accuracy_trend.iter().sum::<f32>() / self.accuracy_trend.len() as f32;
        recent_avg < self.ema_accuracy - 0.05 // 5% decline
    }

    /// Get overall accuracy
    pub fn get_overall_accuracy(&self) -> f32 {
        if self.total_keystrokes == 0 {
            1.0
        } else {
            (self.total_keystrokes - self.total_errors) as f32 / self.total_keystrokes as f32
        }
    }

    /// Get statistics summary
    pub fn get_stats(&self) -> AccuracyStats {
        AccuracyStats {
            overall_accuracy: self.get_overall_accuracy(),
            ema_accuracy: self.ema_accuracy,
            total_keystrokes: self.total_keystrokes,
            total_errors: self.total_errors,
            weak_fingers: self.weak_fingers(),
            is_declining: self.is_declining(),
        }
    }
}

impl Default for AccuracyTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics for accuracy tracking
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct AccuracyStats {
    #[allow(dead_code)]
    pub overall_accuracy: f32,
    #[allow(dead_code)]
    pub ema_accuracy: f32,
    #[allow(dead_code)]
    pub total_keystrokes: u32,
    #[allow(dead_code)]
    pub total_errors: u32,
    #[allow(dead_code)]
    pub weak_fingers: Vec<usize>,
    #[allow(dead_code)]
    pub is_declining: bool,
}
