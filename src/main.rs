//! Ukrainian Keyboard Trainer - LLM-based adaptive typing drills
//!
//! Single-session, stateless, self-contained CLI application.
//! Uses Candle for LLM inference and per-finger tracking.

mod cli;
mod llm;
mod session;

use clap::Parser;
use cli::display::Display;
use cli::input::InputHandler;
use llm::vocab::Vocab;
use session::{AccuracyTracker, ErrorDetector, SessionDecision, SessionState};
use std::error::Error;
use std::fs;
use std::path::Path;

#[derive(Parser, Debug)]
#[command(name = "Ukrainian Keyboard Trainer")]
#[command(about = "Adaptive Ukrainian keyboard typing drills with LLM")]
struct Args {
    /// Training level (1-50)
    #[arg(short, long, default_value = "1")]
    level: u32,

    /// Path to model weights
    #[arg(short, long, default_value = "models/model_weights.bin")]
    model: String,

    /// Path to vocabulary file
    #[arg(short, long, default_value = "data/vocab.json")]
    vocab: String,

    /// Enable debug mode
    #[arg(short, long)]
    debug: bool,
}

/// Load words from the word pool file for a given level
fn load_words(level: u32) -> Result<Vec<String>, Box<dyn Error>> {
    let word_pool_path = format!("data/word_pool/{:02}.txt", level);

    if !Path::new(&word_pool_path).exists() {
        return Err(format!("Word pool file not found: {}", word_pool_path).into());
    }

    let content = fs::read_to_string(&word_pool_path)?;
    let words: Vec<String> = content
        .split_whitespace()
        .map(|w| w.to_string())
        .filter(|w| !w.is_empty())
        .collect();

    if words.is_empty() {
        return Err(format!("No words found in {}", word_pool_path).into());
    }

    Ok(words)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    println!("🇺🇦 Ukrainian Keyboard Trainer v0.1.0");
    println!(
        "Level: {} | Model: {} | Vocab: {}",
        args.level, args.model, args.vocab
    );

    // Initialize display
    let display = Display::simple()?;
    display.clear()?;

    // Load vocabulary
    let _vocab = match Vocab::load(&args.vocab) {
        Ok(v) => {
            if args.debug {
                println!("✓ Vocabulary loaded: {} characters", v.size());
            }
            v
        }
        Err(e) => {
            // Fallback: build from Ukrainian alphabet
            if args.debug {
                eprintln!("⚠ Could not load vocab.json: {} (using default)", e);
            }
            Vocab::new()
        }
    };

    // Initialize session
    let mut session = SessionState::new(args.level);
    session.start();

    // Initialize tracking
    let mut error_detector = ErrorDetector::new();

    // Initialize input handler
    InputHandler::enable_raw_mode()?;
    let input = InputHandler::new();

    // Load words from word pool for this level
    let word_list = load_words(args.level)?;
    let total_words = word_list.len().min(50); // Limit to 50 words per session
    let mut word_index = 0;
    let mut current_word = word_list[word_index].clone();

    let mut user_input = String::new();
    let mut words_completed = 0;

    println!(
        "Loading {} words for Level {} (showing up to 50)...\n",
        word_list.len(),
        args.level
    );

    // Event loop
    'session: loop {
        // Display current state
        display.clear()?;
        display.show_word(&current_word)?;
        display.show_input(&current_word, &user_input)?;

        // Show progress
        let words_until_decision = (10usize).saturating_sub(session.words_since_check as usize);
        display.show_progress(
            words_completed + 1,
            total_words,
            session.total_accuracy,
            words_until_decision,
        )?;

        // Show stats if any words completed
        if words_completed > 0 {
            display.show_finger_stats(&session.per_finger_accuracy)?;
        }

        display.show_help()?;

        // Read input
        match input.read_key()? {
            Some(key) => {
                // Check for exit
                if InputHandler::is_exit(&key) {
                    break 'session;
                }

                // Handle backspace
                if InputHandler::is_backspace(&key) {
                    user_input.pop();
                    continue;
                }

                // Handle enter/submit
                if InputHandler::is_enter(&key) {
                    if !user_input.is_empty() {
                        // Calculate accuracy
                        let (accuracy, _) =
                            AccuracyTracker::calculate_word_accuracy(&current_word, &user_input);

                        // Update session
                        session.record_word(accuracy);
                        words_completed += 1;

                        // Track errors
                        for (target_char, user_char) in current_word.chars().zip(user_input.chars())
                        {
                            if target_char != user_char {
                                error_detector.record_error(target_char, user_char);
                            }
                        }

                        // Show result
                        display.clear()?;
                        display.show_word(&current_word)?;
                        display.show_input(&current_word, &user_input)?;
                        display.show_stats(
                            accuracy,
                            words_completed as u32,
                            session.duration_secs(),
                        )?;

                        // Check decision every 10 words
                        if session.should_check_decision() {
                            let decision = session.make_decision();
                            let reason = match decision {
                                SessionDecision::Next => {
                                    format!(
                                        "Excellent! {}% accuracy with {} words",
                                        (session.total_accuracy * 100.0) as u32,
                                        session.words_typed
                                    )
                                }
                                SessionDecision::Continue => {
                                    let weak = session.weak_fingers();
                                    format!("Keep practicing! Weak fingers: {:?}", weak)
                                }
                                SessionDecision::Reduce => {
                                    format!(
                                        "Accuracy declining to {}%",
                                        (session.total_accuracy * 100.0) as u32
                                    )
                                }
                                SessionDecision::Break => {
                                    format!(
                                        "Session time: {:.1} min or accuracy too low",
                                        session.duration_mins()
                                    )
                                }
                            };

                            let decision_str = match decision {
                                SessionDecision::Next => "next",
                                SessionDecision::Continue => "continue",
                                SessionDecision::Reduce => "reduce",
                                SessionDecision::Break => "break",
                            };

                            display.show_decision(decision_str, &reason)?;

                            if decision == SessionDecision::Break
                                || decision == SessionDecision::Next
                            {
                                println!("Press any key to continue...");
                                let _ = input.read_key()?;
                                break 'session;
                            }

                            session.reset_check_counter();
                        }

                        // Move to next word (cycle through word list)
                        word_index += 1;
                        if word_index >= word_list.len() || words_completed >= total_words {
                            // Reached end of word pool or session limit
                            break 'session;
                        }
                        current_word = word_list[word_index].clone();

                        // Reset for next word
                        user_input.clear();
                    }
                    continue;
                }

                // Add character to input
                if let Some(c) = InputHandler::key_to_char(&key) {
                    user_input.push(c);
                }
            }
            None => {
                // Timeout - just continue
            }
        }
    }

    // Cleanup
    InputHandler::disable_raw_mode()?;
    display.shutdown()?;

    // Summary
    println!("\n🎉 Session Complete!");
    println!(
        "📊 Final Stats: {}% accuracy | {} words | {:.1}s",
        (session.total_accuracy * 100.0) as u32,
        session.words_typed,
        session.duration_secs()
    );

    let error_summary = error_detector.get_error_summary();
    if !error_summary.problematic_chars.is_empty() {
        println!(
            "⚠️  Problematic characters: {:?}",
            error_summary.problematic_chars
        );
    }

    if error_summary.is_improving {
        println!("✅ Error rate improving!");
    }

    println!("🇺🇦 Thanks for practicing!");

    Ok(())
}
