//! Session Management: State tracking, accuracy metrics, and error detection
//!
//! # Components
//! - `state.rs`: SessionState struct for tracking session progress
//! - `accuracy.rs`: Per-finger accuracy tracking with EMA
//! - `errors.rs`: Error pair detection for repeated mistakes

pub mod accuracy;
pub mod errors;
pub mod state;

pub use accuracy::AccuracyTracker;
pub use errors::ErrorDetector;
pub use state::{SessionDecision, SessionState};

// These are only used internally or their fields are accessed directly
#[allow(unused_imports)]
pub use accuracy::AccuracyStats;
#[allow(unused_imports)]
pub use errors::ErrorSummary;
