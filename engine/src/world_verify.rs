//! Self-checks a game reports while it runs: see [`World::check`] and
//! [`World::report`]. A verify run (`ENGINE_VERIFY=<folder>`, see
//! `crate::verify`) writes them into its `report.json` and fails when a
//! check failed.

use super::World;
use serde::Serialize;
use std::collections::BTreeMap;

/// One named check as last reported.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct CheckState {
    pub ok: bool,
    /// How often it was reported, and how often it failed.
    pub runs: u64,
    pub failures: u64,
    /// The frame of the first failure, if any.
    pub first_failure_frame: Option<u64>,
}

#[derive(Debug, Default)]
pub(crate) struct VerifyLog {
    pub frame: u64,
    pub checks: BTreeMap<String, CheckState>,
    pub values: BTreeMap<String, serde_json::Value>,
}

impl World {
    /// Record a self-check: `world.check("player stays above ground", y > 0.0)`.
    ///
    /// Call it every frame (or whenever it matters). Normal runs only keep
    /// count; a verify run (`ENGINE_VERIFY=<folder>`) lists every check in
    /// its report and fails if any check was ever false. Returns `ok`, so it
    /// can sit inside an `if`.
    pub fn check(&mut self, name: impl Into<String>, ok: bool) -> bool {
        let frame = self.verify.frame;
        let state = self.verify.checks.entry(name.into()).or_insert(CheckState {
            ok: true,
            runs: 0,
            failures: 0,
            first_failure_frame: None,
        });
        state.runs += 1;
        state.ok = ok;
        if !ok {
            state.failures += 1;
            state.first_failure_frame.get_or_insert(frame);
        }
        ok
    }

    /// Put a value into the verify report: `world.report("score", score)`.
    ///
    /// The latest value per key is kept (anything serde can serialize:
    /// numbers, strings, vectors, your own structs). Use it to show what the
    /// game state was at the end of a verify run.
    pub fn report(&mut self, key: impl Into<String>, value: impl Serialize) {
        let value = serde_json::to_value(value).unwrap_or_else(|error| {
            serde_json::Value::String(format!("<not serializable: {error}>"))
        });
        self.verify.values.insert(key.into(), value);
    }

    /// Names of the checks that failed at least once so far.
    pub fn failed_checks(&self) -> Vec<String> {
        self.verify
            .checks
            .iter()
            .filter(|(_, c)| c.failures > 0)
            .map(|(name, _)| name.clone())
            .collect()
    }

    pub(crate) fn verify_log(&self) -> &VerifyLog {
        &self.verify
    }

    pub(crate) fn verify_next_frame(&mut self) {
        self.verify.frame += 1;
    }
}
