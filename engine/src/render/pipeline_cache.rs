//! Shader precompilation support: a pipeline cache kept on disk, and
//! pipelines built ahead of time on worker threads.
//!
//! Every render and compute pipeline in the engine is created before the
//! first frame (or, for parts that only some apps use, on a worker thread
//! started at launch), so no frame stops to compile a shader the first time
//! a material or effect appears. Where the driver supports it (Vulkan), the
//! compiled pipelines are also saved to disk, so the next launch skips the
//! driver compile entirely.
//!
//! Settings (environment variables):
//! - `ENGINE_PIPELINE_CACHE=0` turns the disk cache off.
//! - `ENGINE_PIPELINE_CACHE_DIR=<dir>` stores it somewhere else. The default
//!   is the OS cache folder (`%LOCALAPPDATA%\Engine\pipelines` on Windows,
//!   `~/.cache/engine/pipelines` on Linux).
//! - `ENGINE_LOG_PIPELINES=1` prints how long pipeline creation took and
//!   whether the cache was used.

use std::path::PathBuf;
use std::sync::RwLock;
use std::thread::JoinHandle;

/// Caches registered per device, so pipeline code can find the cache for the
/// device it builds on without threading it through every constructor.
static CACHES: RwLock<Vec<(wgpu::Device, wgpu::PipelineCache)>> = RwLock::new(Vec::new());

/// The pipeline cache for `device`, if it has one. Pass the result as the
/// `cache` field of a pipeline descriptor: `cache: get(device).as_ref()`.
pub(crate) fn get(device: &wgpu::Device) -> Option<wgpu::PipelineCache> {
    let caches = CACHES.read().unwrap_or_else(|e| e.into_inner());
    caches
        .iter()
        .find(|(d, _)| d == device)
        .map(|(_, cache)| cache.clone())
}

/// True when `ENGINE_LOG_PIPELINES` asks for timing lines.
pub(crate) fn log_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ENGINE_LOG_PIPELINES")
            .map(|v| !matches!(v.trim(), "" | "0" | "off" | "false"))
            .unwrap_or(false)
    })
}

fn cache_wanted() -> bool {
    std::env::var("ENGINE_PIPELINE_CACHE")
        .map(|v| !matches!(v.trim(), "0" | "off" | "false"))
        .unwrap_or(true)
}

/// The device feature to request so the disk cache can work. Empty when the
/// adapter cannot cache or the cache is turned off.
pub(crate) fn wanted_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    if cache_wanted()
        && adapter.features().contains(wgpu::Features::PIPELINE_CACHE)
        && wgpu::util::pipeline_cache_key(&adapter.get_info()).is_some()
    {
        wgpu::Features::PIPELINE_CACHE
    } else {
        wgpu::Features::empty()
    }
}

fn cache_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("ENGINE_PIPELINE_CACHE_DIR") {
        if !dir.trim().is_empty() {
            return Some(PathBuf::from(dir));
        }
    }
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Caches"))
    } else {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
    }?;
    let engine = if cfg!(windows) { "Engine" } else { "engine" };
    Some(base.join(engine).join("pipelines"))
}

/// A device's pipeline cache plus the file it is saved to.
pub(crate) struct DiskCache {
    device: wgpu::Device,
    cache: wgpu::PipelineCache,
    path: PathBuf,
    /// Size of the data last read or written, to skip saving when unchanged.
    saved_len: usize,
}

impl DiskCache {
    /// Load (or start) the cache for this device. `None` when the device was
    /// created without [`wanted_features`] or no cache folder is known.
    pub(crate) fn open(adapter: &wgpu::Adapter, device: &wgpu::Device) -> Option<Self> {
        if !device.features().contains(wgpu::Features::PIPELINE_CACHE) {
            return None;
        }
        let key = wgpu::util::pipeline_cache_key(&adapter.get_info())?;
        // One file per app, so two games never fight over one cache.
        let app = std::env::current_exe()
            .ok()
            .and_then(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()))
            .unwrap_or_else(|| "app".into());
        let path = cache_dir()?.join(format!("{app}-{key}"));
        let data = std::fs::read(&path).ok();
        // SAFETY: the data is either None or bytes this engine wrote with
        // `get_data` for the same adapter key; `fallback: true` makes wgpu
        // validate the header and start empty if it does not match.
        let cache = unsafe {
            device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("engine-pipeline-cache"),
                data: data.as_deref(),
                fallback: true,
            })
        };
        if log_enabled() {
            match &data {
                Some(d) => eprintln!(
                    "[engine] pipeline cache: loaded {} KiB from {}",
                    d.len() / 1024,
                    path.display()
                ),
                None => eprintln!(
                    "[engine] pipeline cache: none yet, will save to {}",
                    path.display()
                ),
            }
        }
        CACHES
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .push((device.clone(), cache.clone()));
        Some(Self {
            device: device.clone(),
            cache,
            path,
            saved_len: data.map(|d| d.len()).unwrap_or(0),
        })
    }

    /// Write the cache to disk if it grew. Cheap when nothing changed.
    pub(crate) fn save(&mut self) {
        let Some(data) = self.cache.get_data() else {
            return;
        };
        if data.len() == self.saved_len {
            return;
        }
        let Some(dir) = self.path.parent() else {
            return;
        };
        if std::fs::create_dir_all(dir).is_err() {
            return;
        }
        // Write then rename, so a crash never leaves half a file.
        let tmp = self.path.with_extension("tmp");
        if std::fs::write(&tmp, &data).is_ok() && std::fs::rename(&tmp, &self.path).is_ok() {
            if log_enabled() {
                eprintln!(
                    "[engine] pipeline cache: saved {} KiB to {}",
                    data.len() / 1024,
                    self.path.display()
                );
            }
            self.saved_len = data.len();
        }
    }
}

impl Drop for DiskCache {
    fn drop(&mut self) {
        self.save();
        let mut caches = CACHES.write().unwrap_or_else(|e| e.into_inner());
        caches.retain(|(d, _)| d != &self.device);
    }
}

/// Something built on a worker thread at startup and picked up the first
/// time it is needed. If it is needed before the worker finishes, the caller
/// waits only for the rest of that work.
pub(crate) enum Prewarm<T> {
    Pending(JoinHandle<T>),
    Ready(T),
    /// Moved out with [`Prewarm::take`].
    Taken,
}

impl<T: Send + 'static> Prewarm<T> {
    /// Start building `build` on a worker thread now.
    pub(crate) fn spawn(name: &str, build: impl FnOnce() -> T + Send + 'static) -> Self {
        match std::thread::Builder::new()
            .name(format!("engine-prewarm-{name}"))
            .spawn(build)
        {
            Ok(handle) => Self::Pending(handle),
            // No threads (unusual platforms): build later, on first use.
            Err(_) => Self::Taken,
        }
    }

    /// True when the worker has finished (or the value was taken).
    pub(crate) fn is_ready(&self) -> bool {
        match self {
            Self::Pending(h) => h.is_finished(),
            _ => true,
        }
    }

    fn settle(&mut self) {
        if let Self::Pending(_) = self {
            let Self::Pending(handle) = std::mem::replace(self, Self::Taken) else {
                unreachable!()
            };
            match handle.join() {
                Ok(value) => *self = Self::Ready(value),
                Err(panic) => std::panic::resume_unwind(panic),
            }
        }
    }

    /// The built value, waiting for the worker if needed. `None` after
    /// [`Prewarm::take`] or when no worker could be started.
    pub(crate) fn get(&mut self) -> Option<&mut T> {
        self.settle();
        match self {
            Self::Ready(value) => Some(value),
            _ => None,
        }
    }

    /// Move the built value out, waiting for the worker if needed.
    pub(crate) fn take(&mut self) -> Option<T> {
        self.settle();
        match std::mem::replace(self, Self::Taken) {
            Self::Ready(value) => Some(value),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Prewarm;

    #[test]
    fn prewarm_hands_over_the_built_value() {
        let mut warm = Prewarm::spawn("test", || 41 + 1);
        assert_eq!(warm.get().copied(), Some(42));
        assert!(warm.is_ready());
        assert_eq!(warm.take(), Some(42));
        assert_eq!(warm.take(), None, "a taken value is gone");
    }
}
