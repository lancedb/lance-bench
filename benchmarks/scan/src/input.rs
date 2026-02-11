//! Input file format detection and loading.

use anyhow::{Context, Result};
use arrow::record_batch::RecordBatch;
use std::fs::File;
use std::path::Path;

/// Supported input file formats.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InputFormat {
    Csv,
    Parquet,
    Json,
    Lance,
}

impl InputFormat {
    /// Detect format from file extension.
    pub fn from_path(path: &Path) -> Result<Self> {
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .ok_or_else(|| anyhow::anyhow!("File has no extension: {}", path.display()))?;

        match extension.as_str() {
            "csv" => Ok(InputFormat::Csv),
            "parquet" => Ok(InputFormat::Parquet),
            "json" | "jsonl" | "ndjson" => Ok(InputFormat::Json),
            "lance" => Ok(InputFormat::Lance),
            ext => Err(anyhow::anyhow!(
                "Unsupported file extension: '{}'. Supported: csv, parquet, json, jsonl, ndjson, lance",
                ext
            )),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            InputFormat::Csv => "CSV",
            InputFormat::Parquet => "Parquet",
            InputFormat::Json => "JSON",
            InputFormat::Lance => "Lance",
        }
    }
}

/// Load input file into RecordBatches.
pub fn load_input(path: &Path) -> Result<Vec<RecordBatch>> {
    let format = InputFormat::from_path(path)?;
    println!("  Detected format: {}", format.name());

    match format {
        InputFormat::Csv => load_csv(path),
        InputFormat::Parquet => load_parquet(path),
        InputFormat::Json => load_json(path),
        InputFormat::Lance => load_lance(path),
    }
}

fn load_csv(path: &Path) -> Result<Vec<RecordBatch>> {
    use arrow_csv::ReaderBuilder;
    use std::sync::Arc;

    let file = File::open(path).context("Failed to open CSV file")?;

    // Infer schema from first 1000 rows
    let (schema, _) = arrow_csv::reader::Format::default()
        .with_header(true)
        .infer_schema(&file, Some(1000))?;

    // Reopen file and create reader
    let file = File::open(path)?;
    let reader = ReaderBuilder::new(Arc::new(schema))
        .with_header(true)
        .with_batch_size(65536)
        .build(file)?;

    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>()?;
    Ok(batches)
}

fn load_parquet(path: &Path) -> Result<Vec<RecordBatch>> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = File::open(path).context("Failed to open Parquet file")?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>()?;
    Ok(batches)
}

fn load_json(path: &Path) -> Result<Vec<RecordBatch>> {
    use arrow_json::ReaderBuilder;
    use std::io::BufReader;
    use std::sync::Arc;

    let file = File::open(path).context("Failed to open JSON file")?;
    let buf_reader = BufReader::new(&file);

    // Infer schema from first 1000 records - returns (Schema, usize)
    let (schema, _) = arrow_json::reader::infer_json_schema(buf_reader, Some(1000))?;

    // Reopen and read
    let file = File::open(path)?;
    let buf_reader = BufReader::new(file);
    let reader = ReaderBuilder::new(Arc::new(schema))
        .with_batch_size(65536)
        .build(buf_reader)?;

    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>()?;
    Ok(batches)
}

fn load_lance(path: &Path) -> Result<Vec<RecordBatch>> {
    use futures::TryStreamExt;

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    runtime.block_on(async {
        let dataset = lance::dataset::Dataset::open(path.to_str().unwrap())
            .await
            .context("Failed to open Lance dataset")?;

        let stream = dataset.scan().try_into_stream().await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        Ok(batches)
    })
}

/// Calculate total row count from batches.
pub fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.num_rows()).sum()
}

/// Calculate total byte size (approximate) from batches.
pub fn total_bytes(batches: &[RecordBatch]) -> usize {
    batches
        .iter()
        .map(|b| b.get_array_memory_size())
        .sum()
}
