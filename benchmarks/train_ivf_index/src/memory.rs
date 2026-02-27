//! Process resource monitoring utilities.
//!
//! Provides RSS tracking via background thread and point-in-time CPU/RSS reads
//! from /proc/self for Linux systems.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Monitors process RSS in a background thread to capture peak memory usage.
///
/// Uses /proc/self/statm on Linux which correctly captures Rust-side allocations
/// (unlike Python's tracemalloc which only tracks the Python heap).
pub struct PeakRssMonitor {
    stop: Arc<AtomicBool>,
    baseline: u64,
    peak: Arc<AtomicU64>,
    handle: Option<thread::JoinHandle<()>>,
}

impl PeakRssMonitor {
    pub fn new() -> Self {
        Self {
            stop: Arc::new(AtomicBool::new(false)),
            baseline: 0,
            peak: Arc::new(AtomicU64::new(0)),
            handle: None,
        }
    }

    pub fn start(&mut self) {
        let baseline = get_rss_bytes();
        self.baseline = baseline;
        self.peak.store(baseline, Ordering::Relaxed);
        self.stop.store(false, Ordering::Relaxed);

        let stop = self.stop.clone();
        let peak = self.peak.clone();

        self.handle = Some(thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let rss = get_rss_bytes();
                peak.fetch_max(rss, Ordering::Relaxed);
                thread::sleep(Duration::from_millis(50));
            }
        }));
    }

    /// Stop monitoring and return (peak_rss_bytes, delta_rss_bytes).
    pub fn stop(mut self) -> (u64, u64) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle.join().unwrap();
        }
        // One final sample
        let rss = get_rss_bytes();
        self.peak.fetch_max(rss, Ordering::Relaxed);
        let peak = self.peak.load(Ordering::Relaxed);
        (peak, peak.saturating_sub(self.baseline))
    }
}

/// Read the current RSS of this process in bytes from /proc/self/statm.
pub fn get_rss_bytes() -> u64 {
    // /proc/self/statm fields: size resident shared text lib data dt
    // Field at index 1 is RSS in pages.
    let statm = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
    let rss_pages: u64 = statm
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
    rss_pages * page_size
}

/// Read the cumulative CPU time (user + system) of this process in seconds
/// from /proc/self/stat.
pub fn get_cpu_time_secs() -> f64 {
    // /proc/self/stat fields (0-indexed, space-separated):
    //   index 13 = utime  (user mode, clock ticks)
    //   index 14 = stime  (kernel mode, clock ticks)
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let fields: Vec<&str> = stat.split_whitespace().collect();
    if fields.len() > 14 {
        let utime: u64 = fields[13].parse().unwrap_or(0);
        let stime: u64 = fields[14].parse().unwrap_or(0);
        let ticks_per_sec = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as f64;
        (utime + stime) as f64 / ticks_per_sec
    } else {
        0.0
    }
}
