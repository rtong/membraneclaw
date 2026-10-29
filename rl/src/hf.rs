//! Checkpoints in the local Hugging Face cache. Nothing here downloads: a checkpoint is
//! read where `huggingface_hub` left it.

use std::path::PathBuf;

/// The cache's root: `$HF_HUB_CACHE`, else `$HF_HOME/hub`, else
/// `~/.cache/huggingface/hub`.
pub fn hub_cache() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("HF_HUB_CACHE") {
        return Some(dir.into());
    }
    if let Some(dir) = std::env::var_os("HF_HOME") {
        return Some(PathBuf::from(dir).join("hub"));
    }
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/huggingface/hub"))
}

/// The directory holding `repo`'s files (`"Qwen/Qwen3.5-0.8B"`) at the revision the
/// cache's `refs/main` names.
pub fn snapshot(repo: &str) -> Result<PathBuf, String> {
    let cache = hub_cache().ok_or("no Hugging Face cache: set HF_HUB_CACHE or HF_HOME")?;
    let root = cache.join(format!("models--{}", repo.replace('/', "--")));
    let refs = root.join("refs/main");
    let revision = std::fs::read_to_string(&refs)
        .map_err(|e| format!("{repo} is not in {}: {e}", cache.display()))?;
    let dir = root.join("snapshots").join(revision.trim());
    if dir.is_dir() {
        Ok(dir)
    } else {
        Err(format!(
            "{} names a revision with no snapshot",
            refs.display()
        ))
    }
}
