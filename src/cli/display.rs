//! Terminal display and UI rendering
//!
//! Features:
//! - Real-time word display with color coding
//! - Accuracy metrics overlay
//! - Per-finger accuracy breakdown
//! - Session stats

#[allow(unused_imports)]
use crossterm::terminal::{EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::{
    cursor, execute,
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal::{self, ClearType},
};
use std::io::{stdout, Write};

/// Terminal display manager
pub struct Display {
    /// Whether we're using alternate screen
    use_alternate_screen: bool,
}

impl Display {
    /// Create display without alternate screen (simpler mode)
    pub fn simple() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Display {
            use_alternate_screen: false,
        })
    }

    /// Clear screen
    pub fn clear(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut stdout = stdout();
        execute!(
            stdout,
            terminal::Clear(ClearType::All),
            cursor::MoveTo(0, 0)
        )?;
        Ok(())
    }

    /// Render target word with padding
    pub fn show_word(&self, word: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut stdout = stdout();

        execute!(
            stdout,
            cursor::MoveTo(0, 2),
            SetForegroundColor(Color::Cyan),
            Print("Target: "),
            ResetColor,
            Print(word),
            Print("\n")
        )?;
        stdout.flush()?;
        Ok(())
    }

    /// Show user input with character-by-character highlighting
    /// Green for correct, Red for incorrect
    pub fn show_input(
        &self,
        target: &str,
        user_input: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut stdout = stdout();

        execute!(
            stdout,
            cursor::MoveTo(0, 3),
            SetForegroundColor(Color::Yellow),
            Print("Your Input: "),
            ResetColor
        )?;

        let target_chars: Vec<char> = target.chars().collect();
        let user_chars: Vec<char> = user_input.chars().collect();

        // Display with color coding
        for (i, &user_char) in user_chars.iter().enumerate() {
            let target_char = target_chars.get(i);

            if Some(&user_char) == target_char {
                execute!(
                    stdout,
                    SetForegroundColor(Color::Green),
                    Print(user_char),
                    ResetColor
                )?;
            } else {
                execute!(
                    stdout,
                    SetForegroundColor(Color::Red),
                    Print(user_char),
                    ResetColor
                )?;
            }
        }

        // Show remaining target characters (if any)
        if user_chars.len() < target_chars.len() {
            execute!(
                stdout,
                SetForegroundColor(Color::DarkGrey),
                Print(&target[user_input.len()..]),
                ResetColor
            )?;
        }

        execute!(stdout, Print("\n"))?;
        stdout.flush()?;
        Ok(())
    }

    /// Display progress (current/total words and accuracy target)
    pub fn show_progress(
        &self,
        current_word: usize,
        total_words: usize,
        accuracy: f32,
        words_until_decision: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut stdout = stdout();

        execute!(
            stdout,
            cursor::MoveTo(0, 5),
            SetForegroundColor(Color::Magenta),
            Print("Progress: "),
            ResetColor,
            Print(format!("{}/{} words", current_word, total_words)),
            Print("  |  "),
            Print("Accuracy: "),
            SetForegroundColor(if accuracy > 0.9 {
                Color::Green
            } else if accuracy > 0.8 {
                Color::Yellow
            } else {
                Color::Red
            }),
            Print(format!("{:.0}%", accuracy * 100.0)),
            ResetColor,
            Print(format!("  |  Next check: {} words\n", words_until_decision)),
        )?;
        stdout.flush()?;
        Ok(())
    }

    /// Display accuracy metrics
    pub fn show_stats(
        &self,
        accuracy: f32,
        words: u32,
        duration_secs: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut stdout = stdout();
        let wps = if duration_secs > 0.0 {
            words as f64 / duration_secs
        } else {
            0.0
        };

        execute!(
            stdout,
            cursor::MoveTo(0, 5),
            SetForegroundColor(Color::Blue),
            Print("─".repeat(50)),
            Print("\n"),
            ResetColor,
            Print("Accuracy: "),
            SetForegroundColor(if accuracy > 0.9 {
                Color::Green
            } else if accuracy > 0.8 {
                Color::Yellow
            } else {
                Color::Red
            }),
            Print(format!("{:.1}%", accuracy * 100.0)),
            ResetColor,
            Print(format!(
                "  |  Words: {}  |  Time: {:.1}s  |  WPS: {:.2}\n",
                words, duration_secs, wps
            )),
        )?;
        stdout.flush()?;
        Ok(())
    }

    /// Show per-finger accuracy breakdown (8 fingers)
    pub fn show_finger_stats(
        &self,
        per_finger: &[f32; 8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut stdout = stdout();
        let finger_names = ["L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"];

        execute!(
            stdout,
            cursor::MoveTo(0, 6),
            SetForegroundColor(Color::Magenta),
            Print("Per-Finger Accuracy: "),
            ResetColor
        )?;

        for (i, &acc) in per_finger.iter().enumerate() {
            let color = if acc > 0.9 {
                Color::Green
            } else if acc > 0.8 {
                Color::Yellow
            } else {
                Color::Red
            };
            execute!(
                stdout,
                Print(finger_names[i]),
                Print(": "),
                SetForegroundColor(color),
                Print(format!("{:.0}% ", acc * 100.0)),
                ResetColor
            )?;
        }

        execute!(stdout, Print("\n"))?;
        stdout.flush()?;
        Ok(())
    }

    /// Show decision message
    pub fn show_decision(
        &self,
        decision: &str,
        reason: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut stdout = stdout();

        let color = match decision {
            "next" => Color::Green,
            "continue" => Color::Yellow,
            "reduce" => Color::Magenta,
            "break" => Color::Red,
            _ => Color::White,
        };

        execute!(
            stdout,
            cursor::MoveTo(0, 8),
            SetForegroundColor(Color::Blue),
            Print("─".repeat(50)),
            Print("\n"),
            ResetColor,
            SetForegroundColor(color),
            Print("Decision: "),
            Print(decision.to_uppercase()),
            ResetColor,
            Print(format!("\nReason: {}\n", reason))
        )?;
        stdout.flush()?;
        Ok(())
    }

    /// Show help text
    pub fn show_help(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut stdout = stdout();

        execute!(
            stdout,
            cursor::MoveTo(0, 10),
            SetForegroundColor(Color::DarkGrey),
            Print("Press ENTER to submit word  |  Ctrl+C to exit\n"),
            ResetColor
        )?;
        stdout.flush()?;
        Ok(())
    }

    /// Reset terminal state and cleanup
    pub fn shutdown(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut stdout = stdout();

        if self.use_alternate_screen {
            execute!(stdout, LeaveAlternateScreen, cursor::Show,)?;
        }

        terminal::disable_raw_mode()?;
        Ok(())
    }
}

impl Default for Display {
    fn default() -> Self {
        // Return simple display that doesn't use alternate screen
        Display {
            use_alternate_screen: false,
        }
    }
}

impl Drop for Display {
    fn drop(&mut self) {
        // Best effort cleanup
        let _ = self.shutdown();
    }
}
