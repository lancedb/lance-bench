//! Storage engine implementations for scan benchmark.

mod lance;
mod parquet;
mod parquet_async;
mod traits;
mod vortex;

pub use lance::LanceEngine;
pub use parquet::ParquetEngine;
pub use parquet_async::ParquetAsyncEngine;
pub use traits::{EngineRegistry, ScanEngine, ScanHandle};
pub use vortex::VortexEngine;

/// Create a registry with all available engines.
pub fn create_registry() -> EngineRegistry {
    let mut registry = EngineRegistry::new();
    // Lance engines with different data storage versions
    registry.register(std::sync::Arc::new(LanceEngine::v2_0()));
    registry.register(std::sync::Arc::new(LanceEngine::v2_1()));
    registry.register(std::sync::Arc::new(LanceEngine::v2_2()));
    // Parquet engines
    registry.register(std::sync::Arc::new(ParquetEngine::new()));
    registry.register(std::sync::Arc::new(ParquetAsyncEngine::new()));
    // Vortex engine
    registry.register(std::sync::Arc::new(VortexEngine::new()));
    registry
}
