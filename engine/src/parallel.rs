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

/// The threads that share out a frame's own work (animation, recording
/// draws, LOD assignment). Background jobs (world streaming, chunk builds,
/// the game's own workers) never run here, so frame work cannot queue
/// behind them: rayon's global pool may be full of slow builds while the
/// frame waits. Every call joins before it returns (`install`, `par_iter`),
/// so a frame step finishes completely before the next one starts.
pub(crate) fn frame_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        // The calling thread waits while these work. Background workers
        // run at a lower priority, so these win the cores they need.
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        rayon::ThreadPoolBuilder::new()
            .num_threads(cores.clamp(1, 8))
            .thread_name(|i| format!("engine-frame-{i}"))
            .build()
            .expect("build the frame thread pool")
    })
}

/// Run `job` on a frame thread while `main` runs here, and return both
/// results once both are done.
///
/// For work a frame needs every frame beside the calls that must stay on
/// the main thread (the ones that take `&mut World`): stepping a traffic
/// or crowd simulation while the last step's result is handed to the
/// world. The frame threads are already running and idle while the game's
/// closure runs, so `job` starts at once. A thread started for it each
/// frame can wait many milliseconds for a core when background workers
/// keep them all busy, and the frame waits with it.
///
/// ```
/// # use engine::parallel::beside_frame;
/// let mut cars = vec![1.0_f32, 2.0, 3.0];
/// let (furthest, drawn) = beside_frame(
///     || {
///         cars.iter_mut().for_each(|c| *c += 0.5);
///         cars.iter().copied().fold(0.0, f32::max)
///     },
///     || "the main thread's own work",
/// );
/// assert_eq!((furthest, drawn), (3.5, "the main thread's own work"));
/// ```
///
/// Both closures have finished when this returns, so they may borrow from
/// the caller. A panic in either is passed on.
pub fn beside_frame<A, B>(job: impl FnOnce() -> A + Send, main: impl FnOnce() -> B) -> (A, B)
where
    A: Send,
{
    let mut made = None;
    let here = frame_pool().in_place_scope(|scope| {
        scope.spawn(|_| made = Some(job()));
        main()
    });
    // The scope returns only after its job ran; had the job panicked, the
    // scope would have passed the panic on instead.
    (made.expect("the scope waited for its job"), here)
}

/// Mark the calling thread as a background worker: call it first thing in
/// a thread that builds or loads things over many frames (meshes, plans,
/// textures). It lowers the thread's priority, so the frame's own threads
/// always win the cores. Without it a busy machine lets a build hold a
/// core for a whole time slice (up to ~15 ms on Windows) while the frame
/// waits. The engine's own streaming workers are marked the same way.
pub fn background_thread() {
    lower_thread_priority();
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
    fn a_job_beside_the_frame_borrows_and_passes_its_panic_on() {
        let mut total = 0_u64;
        let main_thread = std::thread::current().id();
        let (other, here) = beside_frame(
            || {
                total = (1..=100).sum();
                std::thread::current().id()
            },
            || std::thread::current().id(),
        );
        assert_eq!(total, 5050);
        assert_eq!(here, main_thread, "the main half stays on the caller");
        assert_ne!(other, main_thread, "the job runs on a frame thread");
        let failed = std::panic::catch_unwind(|| beside_frame(|| panic!("job failed"), || 1));
        assert!(failed.is_err());
    }

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
