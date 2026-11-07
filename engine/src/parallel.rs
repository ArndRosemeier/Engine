//! Build several things at once on all cores.
//!
//! Generating trees, rocks and textures is the slow part of starting a
//! scene. `build_all` runs those calls side by side:
//!
//! ```no_run
//! # use engine::prelude::*;
//! # fn f() -> EngineResult<()> {
//! let species = [TreeSpecies::Oak, TreeSpecies::Pine, TreeSpecies::Birch];
//! let trees = build_all(&species, |&s| tree_model(s, 0))?;
//! # Ok(())
//! # }
//! ```

use crate::error::EngineResult;
use rayon::prelude::*;

/// Run `make` for every item on all cores and return the results in the
/// items' order. If any call fails, one of the errors is returned.
pub fn build_all<I, T, F>(items: &[I], make: F) -> EngineResult<Vec<T>>
where
    I: Sync,
    T: Send,
    F: Fn(&I) -> EngineResult<T> + Sync + Send,
{
    items.par_iter().map(make).collect()
}

/// Lower the calling thread's priority, for background workers (world
/// streaming). Frame work then always wins the cores: on a busy machine
/// the OS would otherwise let a streaming job hold a core for a whole time
/// slice (up to ~15 ms on Windows) while the frame waits on it.
pub(crate) fn lower_thread_priority() {
    #[cfg(windows)]
    // SAFETY: changes only the current thread's scheduling priority.
    unsafe {
        use windows_sys::Win32::System::Threading::{
            GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL,
        };
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
    #[cfg(target_os = "linux")]
    // SAFETY: on Linux a thread id names one thread; this sets its nice value.
    unsafe {
        let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
        libc::setpriority(libc::PRIO_PROCESS, tid, 10);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::EngineError;

    #[test]
    fn keeps_order_and_reports_errors() {
        let items: Vec<u32> = (0..100).collect();
        let out = build_all(&items, |&i| Ok(i * 2)).unwrap();
        assert_eq!(out, (0..100).map(|i| i * 2).collect::<Vec<_>>());
        let failed = build_all(&items, |&i| {
            if i == 57 {
                Err(EngineError::InvalidValue("bad".into()))
            } else {
                Ok(i)
            }
        });
        assert!(failed.is_err());
    }
}
