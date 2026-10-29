//! Notebook 07's task — WaterTAP questions answered with tools — as the trainer needs it: the
//! harness that runs an episode ([`episode`]), the tools ([`tools`]), the reward ([`reward`]) and
//! the rows the trainer reads off an episode ([`batch`]). Each is held to 07's own Python on
//! episodes recorded in its runs.

pub mod batch;
pub mod episode;
pub mod pyjson;
pub mod reward;
pub mod tools;
