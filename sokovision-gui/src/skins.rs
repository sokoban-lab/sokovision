//! Where the learned skin library lives and how it reaches the disk.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use sokovision_core::SkinLibrary;

/// Writes the skin library off the UI thread. Writes are serialised and an
/// older snapshot is skipped when a newer one is already queued, so a slow or
/// scanned disk can never freeze the window.
pub(crate) struct SkinSaver {
    path: PathBuf,
    latest: AtomicU64,
    /// Newest generation known to have reached the disk.
    written: AtomicU64,
    writing: Mutex<()>,
}

impl SkinSaver {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            latest: AtomicU64::new(0),
            written: AtomicU64::new(0),
            writing: Mutex::new(()),
        }
    }

    pub fn save(self: &Arc<Self>, text: String) {
        let generation = self.latest.fetch_add(1, Ordering::SeqCst) + 1;
        let saver = Arc::clone(self);
        std::thread::spawn(move || {
            if saver.latest.load(Ordering::SeqCst) != generation {
                return;
            }
            let _guard = saver
                .writing
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if saver.latest.load(Ordering::SeqCst) != generation {
                return;
            }
            if let Some(parent) = saver.path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            if fs::write(&saver.path, text).is_ok() {
                saver.written.store(generation, Ordering::SeqCst);
            }
        });
    }

    /// A queued write that has not reached the disk yet, including one still
    /// running on a worker thread.
    pub fn has_pending_write(&self) -> bool {
        self.written.load(Ordering::SeqCst) < self.latest.load(Ordering::SeqCst)
    }

    /// Blocking variant for shutdown, after invalidating queued snapshots.
    pub fn save_now(&self, text: &str) {
        let generation = self.latest.fetch_add(1, Ordering::SeqCst) + 1;
        let _guard = self
            .writing
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(parent) = self.path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if fs::write(&self.path, text).is_ok() {
            self.written.store(generation, Ordering::SeqCst);
        }
    }
}

pub fn library_path() -> Option<PathBuf> {
    Some(
        dirs::data_local_dir()?
            .join("sokovision")
            .join("skins.json"),
    )
}

/// Missing or unreadable files simply mean nothing has been learned yet.
pub fn load(path: &Path) -> Option<SkinLibrary> {
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}
