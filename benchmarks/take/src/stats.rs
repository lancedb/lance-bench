//! Statistics computation for benchmark results.

pub struct Statistics {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

pub fn compute_statistics(latencies: &[f64]) -> Statistics {
    let n = latencies.len() as f64;
    let mean = latencies.iter().sum::<f64>() / n;

    let variance = latencies.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    let mut sorted = latencies.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let p50 = sorted[(n * 0.50) as usize];
    let p95 = sorted[(n * 0.95) as usize];
    let p99 = sorted[(n * 0.99) as usize];

    Statistics {
        mean,
        std,
        min,
        max,
        p50,
        p95,
        p99,
    }
}
