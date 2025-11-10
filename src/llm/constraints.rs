//! Finger zone constraints: Filter words by finger coverage
//!
//! Uses KBRD finger zone definitions to:
//! - Filter candidate words to stay within target finger zones
//! - Identify weak fingers for focused training

use rustc_hash::FxHashMap;
use std::collections::HashSet;
use std::fs;

/// Constraints engine for finger zone filtering
pub struct Constraints {
    /// Map of zone name to character set
    zones: FxHashMap<String, HashSet<char>>,
}
#[allow(dead_code)]
impl Constraints {
    /// Load finger zone definitions from fingers_config.json
    pub fn from_config(config_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(config_path)?;
        let json: serde_json::Value = serde_json::from_str(&content)?;

        let mut zones = FxHashMap::default();

        if let Some(finger_zones) = json.get("finger_zones").and_then(|v| v.as_object()) {
            for (zone_name, chars_array) in finger_zones {
                let mut char_set = HashSet::new();

                if let Some(chars) = chars_array.as_array() {
                    for char_val in chars {
                        if let Some(char_str) = char_val.as_str() {
                            for c in char_str.chars() {
                                char_set.insert(c);
                            }
                        }
                    }
                }

                zones.insert(zone_name.clone(), char_set);
            }
        }

        Ok(Constraints { zones })
    }

    /// Create new constraints with default zones
    pub fn new() -> Self {
        Constraints {
            zones: FxHashMap::default(),
        }
    }

    /// Get all characters in a zone
    pub fn get_zone_chars(&self, zone: &str) -> Option<Vec<char>> {
        self.zones
            .get(zone)
            .map(|chars| chars.iter().copied().collect::<Vec<_>>())
    }

    /// Filter words to only those using characters from target finger zones
    pub fn filter_by_zones(&self, candidates: &[String], zones: &[&str]) -> Vec<String> {
        candidates
            .iter()
            .filter(|word| self.matches_zones(word, zones))
            .cloned()
            .collect()
    }

    /// Check if a word uses only characters from specific zones
    pub fn matches_zones(&self, word: &str, zones: &[&str]) -> bool {
        if zones.is_empty() {
            return true;
        }

        // Collect all allowed characters from specified zones
        let mut allowed_chars: HashSet<char> = HashSet::new();
        for zone_name in zones {
            if let Some(chars) = self.zones.get(*zone_name) {
                allowed_chars.extend(chars);
            }
        }

        // Check if all characters in word are in allowed set
        word.chars().all(|c| allowed_chars.contains(&c))
    }

    /// Get zone names for a character
    pub fn get_zones_for_char(&self, c: char) -> Vec<String> {
        self.zones
            .iter()
            .filter(|(_, chars)| chars.contains(&c))
            .map(|(zone_name, _)| zone_name.clone())
            .collect()
    }

    /// Identify which zones are used in a word
    pub fn identify_zones(&self, word: &str) -> Vec<String> {
        let mut zones_used = HashSet::new();

        for c in word.chars() {
            let word_zones = self.get_zones_for_char(c);
            zones_used.extend(word_zones);
        }

        let mut result = zones_used.into_iter().collect::<Vec<_>>();
        result.sort();
        result
    }

    /// List all available zones
    pub fn list_zones(&self) -> Vec<String> {
        let mut zones = self.zones.keys().cloned().collect::<Vec<_>>();
        zones.sort();
        zones
    }
}

impl Default for Constraints {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matches_zones() {
        let mut constraints = Constraints::new();
        // Create test zones
        let mut lindex_chars = HashSet::new();
        lindex_chars.insert('а');
        lindex_chars.insert('м');
        constraints.zones.insert("Lindex".to_string(), lindex_chars);

        assert!(constraints.matches_zones("ама", &["Lindex"]));
        assert!(constraints.matches_zones("", &["Lindex"]));
    }

    #[test]
    fn test_identify_zones() {
        let mut constraints = Constraints::new();
        let mut lindex = HashSet::new();
        lindex.insert('а');
        let mut rindex = HashSet::new();
        rindex.insert('о');

        constraints.zones.insert("Lindex".to_string(), lindex);
        constraints.zones.insert("Rindex".to_string(), rindex);

        let zones = constraints.identify_zones("ао");
        assert!(zones.contains(&"Lindex".to_string()));
        assert!(zones.contains(&"Rindex".to_string()));
    }
}
