//! Error detection: Identify error pairs and patterns
//!
//! Detects:
//! - Repeated character mistakes (3+ times)
//! - Common error pairs (e.g., confusing similar keys)
//! - Persistent finger miscalibration

use std::collections::HashMap;

/// Error pair threshold (minimum occurrences to flag as persistent error)
const ERROR_THRESHOLD: u32 = 3;

/// Detects repeated errors and patterns
#[derive(Clone, Debug)]
pub struct ErrorDetector {
    /// Track error pairs (expected char → (got char → count))
    error_pairs: HashMap<char, HashMap<char, u32>>,
    /// Total error events recorded
    total_errors: u32,
    /// Most recent errors (for trending)
    recent_errors: Vec<(char, char)>,
}

#[allow(dead_code)]
impl ErrorDetector {
    /// Create new error detector
    pub fn new() -> Self {
        ErrorDetector {
            error_pairs: HashMap::new(),
            total_errors: 0,
            recent_errors: Vec::with_capacity(50),
        }
    }

    /// Record an error: expected character vs. what was typed
    pub fn record_error(&mut self, expected: char, got: char) {
        if expected == got {
            return; // Not an error
        }

        // Increment error count for this pair
        self.error_pairs
            .entry(expected)
            .or_insert_with(HashMap::new)
            .entry(got)
            .and_modify(|count| *count += 1)
            .or_insert(1);

        self.total_errors += 1;

        // Track recent errors (keep last 50)
        self.recent_errors.push((expected, got));
        if self.recent_errors.len() > 50 {
            self.recent_errors.remove(0);
        }
    }

    /// Get most common error pairs (sorted by frequency)
    pub fn top_error_pairs(&self, count: usize) -> Vec<((char, char), u32)> {
        let mut pairs = Vec::new();

        for (&expected, got_map) in &self.error_pairs {
            for (&got, &error_count) in got_map {
                pairs.push(((expected, got), error_count));
            }
        }

        // Sort by count descending
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs.into_iter().take(count).collect()
    }

    /// Check if a character pair has 3+ errors (persistent error)
    pub fn has_repeated_error(&self, expected: char, got: char) -> bool {
        self.error_pairs
            .get(&expected)
            .and_then(|map| map.get(&got))
            .map(|&count| count >= ERROR_THRESHOLD)
            .unwrap_or(false)
    }

    /// Get error rate for a specific character (how often it's mistyped)
    pub fn get_error_rate(&self, expected: char) -> f32 {
        if let Some(got_map) = self.error_pairs.get(&expected) {
            let total_errors_for_char: u32 = got_map.values().sum();
            (total_errors_for_char as f32) / (total_errors_for_char as f32 + 1.0)
        } else {
            0.0
        }
    }

    /// Get all characters with repeated errors (3+ threshold)
    pub fn get_problematic_chars(&self) -> Vec<char> {
        self.error_pairs
            .iter()
            .filter_map(|(&expected, got_map)| {
                let has_repeated = got_map.values().any(|&count| count >= ERROR_THRESHOLD);
                if has_repeated {
                    Some(expected)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get most common confusion for a specific character
    pub fn get_primary_confusion(&self, expected: char) -> Option<(char, u32)> {
        self.error_pairs.get(&expected).and_then(|got_map| {
            got_map
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(&got, &count)| (got, count))
        })
    }

    /// Check error trend (recent errors vs historical)
    pub fn is_error_improving(&self, window_size: usize) -> bool {
        if self.recent_errors.len() < window_size {
            return true; // Not enough data
        }

        // Compare recent window to older window
        let recent_window = &self.recent_errors[self.recent_errors.len() - window_size..];
        let older_start =
            (self.recent_errors.len() as i32 - 2 * window_size as i32).max(0) as usize;
        let older_window = &self.recent_errors
            [older_start..older_start + window_size.min(self.recent_errors.len() - older_start)];

        if older_window.is_empty() {
            return true;
        }

        // Count unique error pairs in each window
        let mut recent_pairs = std::collections::HashSet::new();
        for &(expected, got) in recent_window {
            recent_pairs.insert((expected, got));
        }

        let mut older_pairs = std::collections::HashSet::new();
        for &(expected, got) in older_window {
            older_pairs.insert((expected, got));
        }

        // Improving if new errors are fewer or different
        recent_pairs.len() <= older_pairs.len()
    }

    /// Get total number of errors recorded
    pub fn total_errors(&self) -> u32 {
        self.total_errors
    }

    /// Get summary of error patterns
    pub fn get_error_summary(&self) -> ErrorSummary {
        let problematic = self.get_problematic_chars();

        ErrorSummary {
            problematic_chars: problematic,
            is_improving: self.is_error_improving(10),
        }
    }

    /// Clear error history (for starting a new drill)
    pub fn reset(&mut self) {
        self.error_pairs.clear();
        self.total_errors = 0;
        self.recent_errors.clear();
    }
}

impl Default for ErrorDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of error patterns
#[derive(Clone, Debug)]
pub struct ErrorSummary {
    pub problematic_chars: Vec<char>,
    pub is_improving: bool,
}
