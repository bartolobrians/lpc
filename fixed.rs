pub struct FixedPredictor;

impl FixedPredictor {
    pub fn best_predictor_order(data: &Vec<i64>) -> Option<u8> {
        let orders = 0..=4;
        let mut best_order = None;
        let mut min_sum_residuals = i64::MAX;

        for order in orders {
            if let Some(residuals) = Self::get_residuals(data, order) {
                let sum_residuals: i64 = residuals.iter().map(|&x| x.abs()).sum();
                if sum_residuals < min_sum_residuals {
                    min_sum_residuals = sum_residuals;
                    best_order = Some(order);
                }
            }
        }

        best_order
    }

    pub fn get_residuals(data: &Vec<i64>, predictor_order: u8) -> Option<Vec<i64>> {
        if predictor_order > 4 || data.len() < (predictor_order as usize) {
            return None;
        }

        let mut residuals = Vec::with_capacity(data.len());

        for i in 0..data.len() {
            let r_i = match predictor_order {
                0 => 0,
                1 => {
                    if i == 0 {
                        return None;
                    }
                    data[i - 1]
                },
                2 => {
                    if i < 2 {
                        return None;
                    }
                    2 * data[i - 1] - data[i - 2]
                },
                3 => {
                    if i < 3 {
                        return None;
                    }
                    3 * data[i - 1] - 3 * data[i - 2] + data[i - 3]
                },
                4 => {
                    if i < 4 {
                        return None;
                    }
                    4 * data[i - 1] - 6 * data[i - 2] + 4 * data[i - 3] - data[i - 4]
                },
                _ => unreachable!(),
            };
            residuals.push(data[i] - r_i);
        }

        Some(residuals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_ietf_02a() {
        let in_vec = vec![
            4302, 7496, 6199, 7427,
            6484, 7436, 6740, 7508,
            6984, 7583, 7182, -5990,
            -6306, -6032, -6299, -6165,
        ];

        let out_vec_ans = vec![
            3194, -1297, 1228,
            -943, 952, -696, 768,
            -524, 599, -401, -13172,
            -316, 274, -267, 134,
        ];

        let ans = FixedPredictor::get_residuals(&in_vec, 1);

        assert!(ans.is_some());
        assert_eq!(ans.unwrap(), out_vec_ans);
    }

    #[test]
    fn best_predictor_order_test() {
        let in_vec = vec![
            4302, 7496, 6199, 7427,
            6484, 7436, 6740, 7508,
            6984, 7583, 7182, -5990,
            -6306, -6032, -6299, -6165,
        ];

        let best_order = FixedPredictor::best_predictor_order(&in_vec);
        assert_eq!(best_order, Some(1));
    }
}
