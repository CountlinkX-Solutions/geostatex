//! A bounded Levenberg-Marquardt for small least-squares problems: a trust region on each
//! parameter's scale, steps that stop short of the bounds, a line search along each step and an
//! active set for parameters held on a bound.

/// Minimises `sum(r(p)^2)` from `p0` within `[lb, ub]` (bounds may be infinite). `scale` is the
/// typical size of each parameter. Returns the parameters and the cost.
pub fn minimise(
    residuals: &dyn Fn(&[f64]) -> Vec<f64>,
    p0: Vec<f64>,
    lb: &[f64],
    ub: &[f64],
    scale: &[f64],
) -> (Vec<f64>, f64) {
    let cost_of = |p: &[f64]| {
        let r = residuals(p);
        if r.iter().any(|v| !v.is_finite()) {
            f64::INFINITY
        } else {
            r.iter().map(|v| v * v).sum::<f64>()
        }
    };
    let clamp = |p: &mut [f64]| {
        for (j, v) in p.iter_mut().enumerate() {
            *v = v.clamp(lb[j], ub[j]);
        }
    };
    let mut p = p0;
    clamp(&mut p);
    let mut cost = cost_of(&p);
    let mut lambda = 1e-3;
    let mut radius: f64 = 0.5;
    let k = p.len();
    for _ in 0..2000 {
        if !cost.is_finite() {
            break;
        }
        let r = residuals(&p);
        let cols: Vec<Vec<f64>> = (0..k)
            .map(|j| {
                let h = 1.49e-8 * p[j].abs().max(1.0);
                let forward = p[j] + h <= ub[j];
                let mut q = p.clone();
                q[j] = if forward { p[j] + h } else { p[j] - h };
                let r1 = residuals(&q);
                let d = if forward { h } else { -h };
                r1.iter().zip(&r).map(|(a, b)| (a - b) / d).collect()
            })
            .collect();
        let grad: Vec<f64> = cols
            .iter()
            .map(|c| c.iter().zip(&r).map(|(a, b)| a * b).sum())
            .collect();
        let free: Vec<usize> = (0..k)
            .filter(|&j| !((p[j] <= lb[j] && grad[j] > 0.0) || (p[j] >= ub[j] && grad[j] < 0.0)))
            .collect();
        if free.is_empty() {
            break;
        }
        let mut improved = false;
        while lambda < 1e16 {
            let a: Vec<Vec<f64>> = free
                .iter()
                .map(|&i| {
                    free.iter()
                        .map(|&j| {
                            let v: f64 = cols[i].iter().zip(&cols[j]).map(|(a, b)| a * b).sum();
                            if i == j {
                                v * (1.0 + lambda) + 1e-300
                            } else {
                                v
                            }
                        })
                        .collect()
                })
                .collect();
            let b: Vec<f64> = free.iter().map(|&j| -grad[j]).collect();
            let Some(step) = solve_small(a, b) else {
                lambda *= 4.0;
                continue;
            };
            // The trust region caps the step on each parameter's scale.
            let mut alpha: f64 = 1.0;
            for (idx, &j) in free.iter().enumerate() {
                let limit = radius * scale[j];
                if step[idx].abs() > limit {
                    alpha = alpha.min(limit / step[idx].abs());
                }
            }
            // Two families of trial points along the step: stopping short of the first bound
            // (0.995 of the way, as scipy's TRF), which keeps a long step from landing on a
            // bound, and projecting onto the bounds, which lets the other parameters move on
            // when one is pinned next to its bound.
            let mut beta = alpha;
            for (idx, &j) in free.iter().enumerate() {
                let d = step[idx];
                let room = if d > 0.0 { ub[j] - p[j] } else { p[j] - lb[j] };
                if d != 0.0 && room > 0.0 && room < d.abs() * beta {
                    beta = 0.995 * room / d.abs();
                }
            }
            let point = |a: f64| {
                let mut q = p.clone();
                for (idx, &j) in free.iter().enumerate() {
                    q[j] += a * step[idx];
                }
                clamp(&mut q);
                q
            };
            let (q, c) = (0..16)
                .flat_map(|t| [beta / f64::powi(2.0, t), alpha / f64::powi(2.0, t)])
                .map(point)
                .map(|q| {
                    let c = cost_of(&q);
                    (q, c)
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .expect("trial steps");
            if c < cost {
                radius = (radius * 2.0).min(4.0);
                let gain = cost - c;
                let moved = q
                    .iter()
                    .zip(&p)
                    .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
                    .fold(0.0, f64::max);
                p = q;
                cost = c;
                lambda = (lambda * 0.3).max(1e-12);
                improved = gain > 1e-15 * cost.max(1e-300) && moved > 1e-14;
                break;
            }
            lambda *= 4.0;
            radius = (radius * 0.5).max(1e-12);
        }
        if !improved {
            break;
        }
    }
    (p, cost)
}

/// Solves a small dense system by Gaussian elimination with partial pivoting.
pub fn solve_small(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in (col + 1)..n {
            let f = a[row][col] / a[col][col];
            let (top, bottom) = a.split_at_mut(row);
            for (x, y) in bottom[0][col..].iter_mut().zip(&top[col][col..]) {
                *x -= f * y;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let s: f64 = ((row + 1)..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - s) / a[row][row];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_a_line_and_respects_bounds() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [1.0, 3.0, 5.0, 7.0];
        let r = |p: &[f64]| x.iter().zip(&y).map(|(x, y)| p[0] * x + p[1] - y).collect();
        let (p, c) = minimise(&r, vec![0.5, 0.5], &[0.0, 0.0], &[10.0, 10.0], &[1.0, 1.0]);
        assert!((p[0] - 2.0).abs() < 1e-8 && (p[1] - 1.0).abs() < 1e-8 && c < 1e-14);
        let (p, _) = minimise(&r, vec![0.5, 0.5], &[0.0, 0.0], &[1.5, 10.0], &[1.0, 1.0]);
        assert!((p[0] - 1.5).abs() < 1e-12);
    }
}
