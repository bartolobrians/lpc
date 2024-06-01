pub struct FixedPredictor;

impl FixedPredictor {
    /// Get order that yields the least sum of residuals
    /// 
    /// The predictor orders are from 0 to 4 inclusive and is retrieved
    /// by finding the predictor that yields the *minimum* absolute
    /// sum of residuals for the given `data` and derived predictor.
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

    /// Get residuals of a fixed predictor order 
    /// 
    /// The predictor orders are from 0 to 4 inclusive and corresponds
    /// to one of the five "fixed" predictor orders written in the FLAC
    /// specification. The predictor orders are defined as follows:
    /// 
    /// 0: r[i] = 0
    /// 1: r[i] = data[i - 1]
    /// 2: r[i] = 2 * data[i - 1] - data[i - 2]
    /// 3: r[i] = 3 * data[i - 1] - 3 * data[i - 2] + data[i - 3]
    /// 4: r[i] = 4 * data[i - 1] - 6 * data[i - 2] + 4 * data[i - 3] - data[i - 4]
    /// 
    /// This function returns a vector with each element containing data[i] - r[i].
    /// 
    /// # Errors
    /// `None` is returned if an error occurs in the function. This includes whether
    /// the predictor order provided is not within 0 and 4 inclusive and whether the
    /// size of `data` is less than the predictor order.
    pub fn get_residuals(data: &Vec<i64>, predictor_order: u8) -> Option<Vec<i64>> {
        if predictor_order > 4 || data.len() <= predictor_order as usize {
            return None;
        }

        let mut residuals = Vec::with_capacity(data.len() - predictor_order as usize);

        for i in predictor_order as usize..data.len() {
            let r_i = match predictor_order {
                0 => 0,
                1 => data[i - 1],
                2 => 2 * data[i - 1] - data[i - 2],
                3 => 3 * data[i - 1] - 3 * data[i - 2] + data[i - 3],
                4 => 4 * data[i - 1] - 6 * data[i - 2] + 4 * data[i - 3] - data[i - 4],
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
    fn best_predictor_order() {
        let in_vec = vec![
            4302, 7496, 6199, 7427,
            6484, 7436, 6740, 7508,
            6984, 7583, 7182, -5990,
            -6306, -6032, -6299, -6165,
        ];

        let best_order = FixedPredictor::best_predictor_order(&in_vec);
        assert_eq!(best_order, Some(1));
    }

    #[test]
    fn empty_data() {
        let in_vec = vec![];
        let ans = FixedPredictor::get_residuals(&in_vec, 1);
        assert!(ans.is_none());

        let best_order = FixedPredictor::best_predictor_order(&in_vec);
        assert!(best_order.is_none());
    }

    #[test]
    fn insufficient_data() {
        let in_vec = vec![1, 2];
        let ans = FixedPredictor::get_residuals(&in_vec, 3);
        assert!(ans.is_none());

        let best_order = FixedPredictor::best_predictor_order(&in_vec);
        assert_eq!(best_order, Some(0));
    }

    #[test]
    fn identical_elements() {
        let in_vec = vec![5, 5, 5, 5, 5];
        let expected_residuals = vec![0, 0, 0, 0];

        for order in 0..=4 {
            let ans = FixedPredictor::get_residuals(&in_vec, order);
            assert!(ans.is_some());
            assert_eq!(ans.unwrap(), expected_residuals);
        }

        let best_order = FixedPredictor::best_predictor_order(&in_vec);
        assert_eq!(best_order, Some(0));
    }

    #[test]
    fn negative_values() {
        let in_vec = vec![-1, -2, -3, -4, -5];
        let expected_residuals_order_1 = vec![-1, -1, -1, -1];
        let expected_residuals_order_2 = vec![-1, -1, -1];
        let expected_residuals_order_3 = vec![-1, -1];
        let expected_residuals_order_4 = vec![-1];

        let ans = FixedPredictor::get_residuals(&in_vec, 1);
        assert!(ans.is_some());
        assert_eq!(ans.unwrap(), expected_residuals_order_1);

        let ans = FixedPredictor::get_residuals(&in_vec, 2);
        assert!(ans.is_some());
        assert_eq!(ans.unwrap(), expected_residuals_order_2);

        let ans = FixedPredictor::get_residuals(&in_vec, 3);
        assert!(ans.is_some());
        assert_eq!(ans.unwrap(), expected_residuals_order_3);

        let ans = FixedPredictor::get_residuals(&in_vec, 4);
        assert!(ans.is_some());
        assert_eq!(ans.unwrap(), expected_residuals_order_4);

        let best_order = FixedPredictor::best_predictor_order(&in_vec);
        assert_eq!(best_order, Some(1));
    }

    #[test]
    fn predictor_orders() {
        let in_vec = vec![1, 2, 4, 8, 16];

        let expected_residuals_order_0 = vec![1, 2, 4, 8, 16];
        let expected_residuals_order_1 = vec![1, 1, 2, 4];
        let expected_residuals_order_2 = vec![1, 0, 0];
        let expected_residuals_order_3 = vec![1, 0];
        let expected_residuals_order_4 = vec![1];

        let ans = FixedPredictor::get_residuals(&in_vec, 0);
        assert!(ans.is_some());
        assert_eq!(ans.unwrap(), expected_residuals_order_0);

        let ans = FixedPredictor::get_residuals(&in_vec, 1);
        assert!(ans.is_some());
        assert_eq!(ans.unwrap(), expected_residuals_order_1);

        let ans = FixedPredictor::get_residuals(&in_vec, 2);
        assert!(ans.is_some());
        assert_eq!(ans.unwrap(), expected_residuals_order_2);

        let ans = FixedPredictor::get_residuals(&in_vec, 3);
        assert!(ans.is_some());
        assert_eq!(ans.unwrap(), expected_residuals_order_3);

        let ans = FixedPredictor::get_residuals(&in_vec, 4);
        assert!(ans.is_some());
        assert_eq!(ans.unwrap(), expected_residuals_order_4);
    }
}

