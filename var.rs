pub struct VarPredictor;

impl VarPredictor {
    /// Get the autocorrelation of a vector of samples
    ///
    /// The function computes the autocorrelations of the provided vector of
    /// data from `R[0]` until `R[max_lag]`. For example, if `max_lag` is 2, then
    /// the output contains three elements corresponding to R[0] until R[2],
    /// respectively.
    pub fn get_autocorrelation(samples: &Vec<i64>, max_lag: u8) -> Vec<f64> {
        let mut autoc = vec![0.0; max_lag as usize + 1];
        for lag in 0..=max_lag {
            let lag = lag as usize;
            for i in lag..samples.len() {
                autoc[lag] += (samples[i] as f64) * (samples[i - lag] as f64);
            }
        }
        autoc
    }

    /// Get the predictor coefficients
    ///
    /// `autoc` contains the autocorrelation vector where `autoc[i]` corresponds to
    /// the autocorrelation value of lag `i - 1`. `predictor_order` should be
    /// less than `autoc.len()`. The coefficients are computed using the Levinson-Durbin
    /// algorithm.
    pub fn get_predictor_coeffs(autoc: &Vec<f64>, predictor_order: u8) -> Vec<f64> {
        let mut coeffs = vec![0.0; predictor_order as usize];
        let mut error = autoc[0];

        for i in 1..=predictor_order as usize {
            let mut k = autoc[i];
            for j in 0..i - 1 {
                k -= coeffs[j] * autoc[i - j - 1];
            }
            k /= error;

            coeffs[i - 1] = k;
            for j in 0..i - 1 {
                coeffs[j] -= k * coeffs[i - j - 2];
            }

            error *= 1.0 - k * k;
        }
        coeffs
    }

    /// Quantize the predictor coefficients and find their shift factor
    ///
    /// The shift factor `S` is computed from the maximum absolute value of a coefficient
    /// `L_max`. This value is computed as `precision - lg(L_max)` or to
    /// the maximum shift value of 1 << 5 = 31, whichever is smaller. Note that it is
    /// possible for this shift factor to be negative. In that case, the shift value
    /// will still be used in quantizing the coefficients but its effective value
    /// will be zero.
    ///
    /// Quantization involves converting the provided floating-point coefficients
    /// into integers. Each of the values are rounded up or down depending on
    /// some accumulated rounding error `\epsilon`. Initially, this error is zero.
    /// For each coefficient `L_i`, the coefficient is multiplied (for positive shift)
    /// or divided (for negative shift) by `1 << abs(S)` to get the raw value `L_i_r + \epsilon`.
    /// Then, `L_i_r + \epsilon` is rounded away from zero to get the quantized coefficient.
    /// The new rounding error `\epsilon = L_i_r + \epsilon - round(L_i_r)` is then updated for the
    /// next coefficient.
    pub fn quantize_coeffs(lpc_coefs: &Vec<f64>, precision: u8) -> (Vec<i64>, u8) {
        let max_coef = lpc_coefs.iter().cloned().fold(f64::MIN, f64::max);
        let shift_factor = (precision as f64 - max_coef.log2()).min(31.0) as i64;
        let mut quantized_coeffs = Vec::new();
        let mut error = 0.0;

        for coef in lpc_coefs {
            let scaled_coef = coef * (1 << shift_factor) as f64;
            let quantized_coef = (scaled_coef + error).round();
            error += scaled_coef - quantized_coef;
            quantized_coeffs.push(quantized_coef as i64);
        }

        (quantized_coeffs, shift_factor as u8)
    }

    /// Compute the residuals from a given linear predictor
    ///
    /// The resulting vector `residual[i]` corresponds to the `i + predictor_order`th
    /// signal. The first `predictor_order` values of the residual are the "warm-up"
    /// samples, or the unencoded samples, equivalent to `&samples[..predictor_order]`.
    ///
    /// The residuals are computed with the `samples` reversed. For some `i`th residual,
    /// `residual[i] = data[i] - (sum(dot(qlp_coefs, samples[i..(i - predictor_order)])) >> qlp_shift)`.
    pub fn get_residuals(samples: &Vec<i64>, qlp_coefs: &Vec<i64>, predictor_order: u8, qlp_shift: u8) -> Vec<i64> {
        let mut residuals = Vec::with_capacity(samples.len());
        let predictor_order = predictor_order as usize;
        let qlp_shift = qlp_shift as usize;

        // Warm-up samples
        for i in 0..predictor_order {
            residuals.push(samples[i]);
        }

        // Compute residuals
        for i in predictor_order..samples.len() {
            let mut prediction = 0;
            for j in 0..predictor_order {
                prediction += qlp_coefs[j] * samples[i - j - 1];
            }
            prediction >>= qlp_shift;
            residuals.push(samples[i] - prediction);
        }

        residuals
    }

    pub fn get_predictor_coeffs_from_samples(samples: &Vec<i64>, predictor_order: u8, bps: u8, block_size: u64) -> (Vec<i64>, u8, u8) {
        let autoc = Self::get_autocorrelation(samples, predictor_order);
        let coeffs = Self::get_predictor_coeffs(&autoc, predictor_order);
        let precision = Self::get_best_precision(bps, block_size);
        let (quantized_coeffs, shift) = Self::quantize_coeffs(&coeffs, precision);

        (quantized_coeffs, precision, shift)
    }

    pub fn get_best_lpc(samples: &Vec<i64>, bps: u8, block_size: u64) -> (Vec<i64>, u8, u8) {
        let max_order = 32;
        let mut min_residual_sum = i64::MAX;
        let mut best_coeffs = Vec::new();
        let mut best_precision = 0;
        let mut best_shift = 0;

        for order in 1..=max_order {
            let (coeffs, precision, shift) = Self::get_predictor_coeffs_from_samples(samples, order, bps, block_size);
            let residuals = Self::get_residuals(samples, &coeffs, order, shift);
            let residual_sum: i64 = residuals.iter().map(|&x| x.abs()).sum();
            if residual_sum < min_residual_sum {
                min_residual_sum = residual_sum;
                best_coeffs = coeffs;
                best_precision = precision;
                best_shift = shift;
            }
        }

        (best_coeffs, best_precision, best_shift)
    }

    pub fn get_best_precision(bps: u8, block_size: u64) -> u8 {
        match bps {
            b if b < 16 => std::cmp::max(1, 2 + bps / 2),
            16 => match block_size {
                192 => 7,
                384 => 8,
                576 => 9,
                1152 => 10,
                2304 => 11,
                4608 => 12,
                _ => 13,
            },
            _ => match block_size {
                384 => 12,
                1152 => 13,
                _ => 14,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_ietf_02() {
        let in_vec = vec![
            0, 79, 111, 78,
            8, -61, -90, -68,
            -13, 42, 67, 53,
            13, -27, -46, -38,
            -12, 14, 24, 19,
            6, -4, -5, 0,
        ];

        let out_vec_ans = vec![
            0, 79, 111, 3,
            -1, -13, -10,
            -6, 2, 8, 8,
            6, 0, -3, -5,
            -4, -1, 1, 1,
            4, 2, 2, 2,
            0,
        ];

        let out_vec = VarPredictor::get_residuals(&in_vec, &vec![7, -6, 2], 3, 2);

        assert_eq!(out_vec_ans, out_vec);
    }
}
