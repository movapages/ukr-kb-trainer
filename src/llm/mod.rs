//! LLM Module: Model inference, scoring, and vocabulary management
//!
//! # Components
//! - `model.rs`: Candle model loading and inference
//! - `scoring.rs`: ScoreFusion logic (LM + frequency + length + finger rules)
//! - `vocab.rs`: Tokenizer (char-level Ukrainian)
//! - `constraints.rs`: Finger-zone filtering

pub mod constraints;
pub mod model;
pub mod scoring;
pub mod vocab;

// pub use constraints::Constraints;
// pub use model::Model;
// pub use scoring::ScoreFusion;
// pub use vocab::Vocab;
