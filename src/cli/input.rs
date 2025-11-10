//! Keystroke input handling using crossterm
//!
//! Features:
//! - Non-blocking keystroke capture
//! - Unicode character support (Ukrainian)
//! - Ctrl+C graceful exit

use crossterm::event::{self, KeyCode, KeyEvent, KeyModifiers};
use std::io::Result as IoResult;
use std::time::Duration;

/// Handles user input from terminal
pub struct InputHandler {
    /// Timeout for poll operations (milliseconds)
    poll_timeout: Duration,
}

#[allow(dead_code)]
impl InputHandler {
    /// Create new input handler with default timeout (50ms for responsive input)
    pub fn new() -> Self {
        InputHandler {
            poll_timeout: Duration::from_millis(50),
        }
    }

    /// Enable raw mode for terminal input
    pub fn enable_raw_mode() -> IoResult<()> {
        crossterm::terminal::enable_raw_mode()
    }

    /// Disable raw mode and restore terminal
    pub fn disable_raw_mode() -> IoResult<()> {
        crossterm::terminal::disable_raw_mode()
    }

    /// Poll for keystroke with timeout (non-blocking)
    /// Returns Some(KeyEvent) if key pressed, None if timeout
    pub fn read_key(&self) -> Result<Option<KeyEvent>, Box<dyn std::error::Error>> {
        if event::poll(self.poll_timeout)? {
            match event::read()? {
                event::Event::Key(key_event) => Ok(Some(key_event)),
                _ => Ok(None),
            }
        } else {
            Ok(None)
        }
    }

    /// Check if key event is an exit signal (Ctrl+C or Escape)
    pub fn is_exit(key: &KeyEvent) -> bool {
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => true,
            KeyCode::Esc => true,
            _ => false,
        }
    }

    /// Convert key event to character (Ukrainian support)
    pub fn key_to_char(key: &KeyEvent) -> Option<char> {
        match key.code {
            // Regular character input (including space which is KeyCode::Char(' '))
            KeyCode::Char(c) => {
                // Only return if no special modifiers (not Ctrl, not Alt)
                if !key.modifiers.contains(KeyModifiers::CONTROL)
                    && !key.modifiers.contains(KeyModifiers::ALT)
                {
                    Some(c)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Check if key is backspace
    pub fn is_backspace(key: &KeyEvent) -> bool {
        matches!(key.code, KeyCode::Backspace)
    }

    /// Check if key is enter/return
    pub fn is_enter(key: &KeyEvent) -> bool {
        matches!(key.code, KeyCode::Enter)
    }
}

impl Default for InputHandler {
    fn default() -> Self {
        Self::new()
    }
}
