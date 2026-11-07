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
