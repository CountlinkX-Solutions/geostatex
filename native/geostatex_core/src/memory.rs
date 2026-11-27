//! Guards before the allocations that grow with the square of the number of points. A failed
//! allocation aborts the process in Rust (the whole BEAM for a NIF), so these checks return an
//! error first when the memory is clearly not there.

/// Copies of an `n`-square matrix alive while a kriging system is solved with LU (measured: the
/// peak of global kriging is about 4.75 matrices at 5000 and 10000 points).
const SOLVE_COPIES: f64 = 5.0;

/// Copies alive while a pseudo-inverse is computed (the decomposition's factors, the inverse and
/// the matrix).
const PSEUDO_INVERSE_COPIES: f64 = 8.0;

/// Bytes of memory available for new allocations: `MemAvailable` of `/proc/meminfo` on Linux,
/// `None` where it is not known.
pub fn available() -> Option<f64> {
    let info = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb = info.lines().find_map(|l| l.strip_prefix("MemAvailable:"))?;
    let kb: f64 = kb.trim().strip_suffix("kB")?.trim().parse().ok()?;
    Some(kb * 1024.0)
}

/// Bytes needed to solve one `size`-square kriging system, by LU or with a pseudo-inverse.
pub fn system_bytes(size: usize, pseudo_inverse: bool) -> f64 {
    let copies = if pseudo_inverse {
        PSEUDO_INVERSE_COPIES
    } else {
        SOLVE_COPIES
    };
    copies * 8.0 * (size as f64) * (size as f64)
}

/// `Ok` when `bytes` fit in the available memory (Linux), or can be reserved where the available
/// memory is not known (the reservation is released at once); otherwise an error saying that
/// `what` needs them.
pub fn ensure(bytes: f64, what: impl FnOnce() -> String) -> Result<(), String> {
    fitting(bytes, what).map(|_| ())
}

/// How many blocks of `bytes` fit at once (at least 1, `usize::MAX` where the available memory
/// is not known and one block can be reserved), or an error saying that `what` needs them.
pub fn fitting(bytes: f64, what: impl FnOnce() -> String) -> Result<usize, String> {
    let gb = |b: f64| format!("{:.1} GB", b / 1e9);
    match available() {
        Some(free) if bytes <= free => Ok((free / bytes.max(1.0)) as usize),
        Some(free) => Err(format!(
            "{} needs about {} of memory, more than the {} available",
            what(),
            gb(bytes),
            gb(free)
        )),
        None if bytes <= isize::MAX as f64
            && Vec::<u8>::new().try_reserve_exact(bytes as usize).is_ok() =>
        {
            Ok(usize::MAX)
        }
        None => Err(format!(
            "{} needs about {} of memory, more than can be reserved",
            what(),
            gb(bytes)
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_cannot_be_allocated_is_refused() {
        let err = ensure(1e30, || "an absurd table".to_string()).unwrap_err();
        assert!(
            err.starts_with("an absurd table needs about") && err.contains("GB"),
            "{err}"
        );
        assert_eq!(ensure(1024.0, || "a small table".to_string()), Ok(()));
        assert_eq!(
            fitting(1024.0, || "small systems".to_string()).map(|k| k >= 1),
            Ok(true)
        );
        assert!(fitting(1e30, || "absurd systems".to_string()).is_err());
    }

    #[test]
    fn a_pseudo_inverse_needs_more_copies_than_a_solve() {
        assert!(system_bytes(1000, false) < system_bytes(1000, true));
        assert_eq!(system_bytes(1000, false), 5.0 * 8.0 * 1e6);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_available_memory_is_known_on_linux() {
        assert!(available().is_some_and(|b| b > 0.0));
    }
}
