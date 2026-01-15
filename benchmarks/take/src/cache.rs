//! Cache management utilities for dropping files from the kernel page cache.

use anyhow::Result;
use std::fs;
use std::path::Path;

pub fn drop_file_cache(file_path: &Path) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;

        const POSIX_FADV_DONTNEED: libc::c_int = 4;

        let file = fs::File::open(file_path)?;
        let fd = file.as_raw_fd();
        let metadata = file.metadata()?;
        let file_size = metadata.len() as i64;

        unsafe {
            libc::posix_fadvise(fd, 0, file_size, POSIX_FADV_DONTNEED);
        }
    }

    Ok(())
}

pub fn drop_directory_cache(path: &Path) -> Result<()> {
    if !path.exists() {
        println!("    Warning: Path does not exist: {}", path.display());
        return Ok(());
    }

    let mut file_count = 0;
    let mut total_size = 0u64;

    for entry in walkdir::WalkDir::new(path) {
        let entry = entry?;
        if entry.file_type().is_file() {
            if let Ok(metadata) = entry.metadata() {
                total_size += metadata.len();
                let _ = drop_file_cache(entry.path());
                file_count += 1;
            }
        }
    }

    println!(
        "    Dropped {} files ({:.2} GB) from cache",
        file_count,
        total_size as f64 / 1024.0 / 1024.0 / 1024.0
    );

    Ok(())
}
