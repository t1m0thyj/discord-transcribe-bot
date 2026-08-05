mod audio;
mod decoder;
mod frontend;
mod models;
mod pipeline;

pub(crate) use models::{resolve_model_dir, validate_model_layout};

pub use audio::{
    ClientDisconnectHandler, SpeakingUpdateHandler, VoiceTickHandler,
    clear_unknown_ssrc_audio_for_guild, decode_queue_capacity, decode_queue_depth,
};
pub use pipeline::{
    AsrEngine, SsrcMap, Streams, should_dispatch_chunk, transcribe_mono_pcm,
    trim_finalize_tail,
};
