import numpy as np

from image_search.metrics import hit_rate


class TestHitRate:
    def test_all_values_first_return_one(self):
        true_ids = np.array([[1, 2]])
        predicted_ids = np.array([
            [1, 20, 30],
            [2, 40, 50],
        ])

        actual = hit_rate(true_ids=true_ids, predicted_ids=predicted_ids, k=[1, 2])
        expected = {1: 1.0, 2: 1.0}

        assert actual == expected

    def test_one_values_second_return_expected(self):
        true_ids = np.array([[1, 2]])
        predicted_ids = np.array([
            [1, 20, 30],
            [40, 2, 50],
        ])

        actual = hit_rate(true_ids=true_ids, predicted_ids=predicted_ids, k=[1, 2])
        expected = {1: 0.5, 2: 1.0}

        assert actual == expected

    def test_no_values_return_zero(self):
        true_ids = np.array([[1, 2]])
        predicted_ids = np.array([
            [10, 20, 30],
            [40, 50, 50],
        ])

        actual = hit_rate(true_ids=true_ids, predicted_ids=predicted_ids, k=[1, 2])
        expected = {1: 0.0, 2: 0.0}

        assert actual == expected
