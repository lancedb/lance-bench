//! Storage engine implementations.

mod lance;
mod parquet;
mod traits;
mod vortex;

pub use lance::LanceEngine;
pub use parquet::ParquetEngine;
pub use traits::{DatasetHandle, EngineRegistry};
pub use vortex::VortexEngine;

/// Create a registry with all available engines.
pub fn create_registry() -> EngineRegistry {
    let mut registry = EngineRegistry::new();
    registry.register(std::sync::Arc::new(LanceEngine::new()));
    registry.register(std::sync::Arc::new(ParquetEngine::new()));
    registry.register(std::sync::Arc::new(VortexEngine::new()));
    registry
}
