//! Session state tracking
//!
//! Maintains:
//! - Current level and word pool
//! - Cumulative accuracy and speed metrics
//! - Per-finger accuracy breakdown
//! - Decision state (continue, reduce, break, next)

use std::time::Instant;

/// Session decision outcome
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SessionDecision {
    /// Continue at current level (>95% accuracy, weak finger detected)
    Continue,
    /// Reduce difficulty (<85% accuracy or declining)
    Reduce,
    /// Break session (<75% accuracy or >40 min)
    Break,
    /// Advance to next level (>90% accuracy + >50 words)
    Next,
}

/// Complete session state
#[derive(Clone, Debug)]
pub struct SessionState {
    /// Total words typed in session
    pub words_typed: u32,
    /// Total accuracy (0.0-1.0)
    pub total_accuracy: f32,
    /// Session start time
    pub start_time: Option<Instant>,
    /// Per-finger accuracy (8 fingers: L1, L2, L3, L4, R1, R2, R3, R4)
    pub per_finger_accuracy: [f32; 8],
    /// Per-finger word counts (for averaging)
    pub per_finger_counts: [u32; 8],
    /// EMA accuracy (exponential moving average)
    pub ema_accuracy: f32,
    /// EMA decay factor (alpha = 0.1)
    pub ema_alpha: f32,
    /// Words since last decision check
    pub words_since_check: u32,
}

impl SessionState {
    /// Create new session for a level
    pub fn new(_level: u32) -> Self {
        SessionState {
            words_typed: 0,
            total_accuracy: 1.0,
            start_time: None,
            per_finger_accuracy: [1.0; 8],
            per_finger_counts: [0; 8],
            ema_accuracy: 1.0,
            ema_alpha: 0.1,
            words_since_check: 0,
        }
    }

    /// Start the session timer
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Get session duration in seconds
    pub fn duration_secs(&self) -> f64 {
        self.start_time
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0)
    }

    /// Get session duration in minutes
    pub fn duration_mins(&self) -> f64 {
        self.duration_secs() / 60.0
    }

    /// Update per-finger accuracy with a new measurement
    #[allow(dead_code)]
    pub fn update_finger_accuracy(&mut self, finger_idx: usize, accuracy: f32) {
        if finger_idx < 8 {
            let count = self.per_finger_counts[finger_idx];
            let new_avg = (self.per_finger_accuracy[finger_idx] * count as f32 + accuracy)
                / (count as f32 + 1.0);
            self.per_finger_accuracy[finger_idx] = new_avg;
            self.per_finger_counts[finger_idx] += 1;
        }
    }

    /// Update EMA accuracy
    pub fn update_ema(&mut self, new_accuracy: f32) {
        self.ema_accuracy =
            self.ema_alpha * new_accuracy + (1.0 - self.ema_alpha) * self.ema_accuracy;
    }

    /// Record a completed word
    pub fn record_word(&mut self, accuracy: f32) {
        self.words_typed += 1;
        self.words_since_check += 1;

        // Update total accuracy as running average
        self.total_accuracy = (self.total_accuracy * (self.words_typed - 1) as f32 + accuracy)
            / self.words_typed as f32;

        // Update EMA
        self.update_ema(accuracy);
    }

    /// Get weak fingers (< 80% accuracy)
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

    /// Decide whether to continue, reduce, break, or advance
    pub fn make_decision(&self) -> SessionDecision {
        let duration = self.duration_mins();

        // Break if too long or accuracy too low
        if duration > 40.0 || self.total_accuracy < 0.75 {
            return SessionDecision::Break;
        }

        // Reduce if accuracy declining or below threshold
        if self.total_accuracy < 0.85 || self.ema_accuracy < 0.85 {
            return SessionDecision::Reduce;
        }

        // Next level if enough words typed and accuracy is good
        if self.words_typed >= 50 && self.total_accuracy > 0.90 {
            return SessionDecision::Next;
        }

        // Continue if excellent accuracy (isolate weak fingers)
        if self.total_accuracy > 0.95 && !self.weak_fingers().is_empty() {
            return SessionDecision::Continue;
        }

        // Default: continue
        SessionDecision::Continue
    }

    /// Reset decision check counter
    pub fn reset_check_counter(&mut self) {
        self.words_since_check = 0;
    }

    /// Check if decision should be evaluated (every ~10 words)
    pub fn should_check_decision(&self) -> bool {
        self.words_since_check >= 10
    }
}
