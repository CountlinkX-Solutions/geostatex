//! Thin Rustler layer over `geostatex_core`. NIFs return `Result<_, String>`; heavy ones run on
//! the dirty CPU scheduler.

mod cokriging;
mod kriging;
mod simulation;
mod variogram;

use rustler::Binary;

#[rustler::nif]
fn nif_version() -> u32 {
    geostatex_core::abi_version()
}

#[rustler::nif(schedule = "DirtyCpu")]
fn sum_f64(binary: Binary) -> Result<f64, String> {
    let values = geostatex_core::bytes_to_f64s(binary.as_slice())?;
    Ok(geostatex_core::sum_f64(&values))
}

rustler::init!("Elixir.Geostatex.Native");
