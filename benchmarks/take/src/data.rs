//! Common data generation utilities for benchmarks.

use arrow::array::{FixedSizeListArray, Float32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use std::sync::Arc;

/// Creates the schema for the vector dataset.
pub fn create_schema(dim: usize) -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
        ),
        true,
    )]))
}

/// Generates a batch of random vectors.
pub fn generate_vector_batch(
    schema: Arc<Schema>,
    batch_size: usize,
    dim: usize,
) -> Result<RecordBatch, arrow::error::ArrowError> {
    let mut rng = rand::thread_rng();
    let mut values: Vec<f32> = Vec::with_capacity(batch_size * dim);
    for _ in 0..batch_size * dim {
        values.push(StandardNormal.sample(&mut rng));
    }
    let values_array = Float32Array::from(values);
    let list_array = FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
        Arc::new(values_array),
        None,
    );

    RecordBatch::try_new(schema, vec![Arc::new(list_array)])
}

/// Generates random query indices.
pub fn generate_queries(num_queries: usize, rows_per_query: usize, max_row: usize) -> Vec<Vec<u64>> {
    let mut rng = rand::thread_rng();
    let mut queries = Vec::with_capacity(num_queries);

    for _ in 0..num_queries {
        let mut query = Vec::with_capacity(rows_per_query);
        for _ in 0..rows_per_query {
            query.push(rng.gen_range(0..max_row as u64));
        }
        query.sort_unstable();
        queries.push(query);
    }

    queries
}
