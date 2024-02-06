import numpy as np
import pytest

from image_search.metrics import hit_rate, mean_average_precision, mean_average_precision_at_k


class TestHitRate:
    def test_all_values_first_return_one(self):
        true_ids = np.array([[1, 2]])
        predicted_ids = np.array(
            [
                [1, 20, 30],
                [2, 40, 50],
            ]
        )

        actual = hit_rate(true_ids=true_ids, predicted_ids=predicted_ids, k=[1, 2])
        expected = {1: 1.0, 2: 1.0}

        assert actual == expected

    def test_one_values_second_return_expected(self):
        true_ids = np.array([[1, 2]])
        predicted_ids = np.array(
            [
                [1, 20, 30],
                [40, 2, 50],
            ]
        )

        actual = hit_rate(true_ids=true_ids, predicted_ids=predicted_ids, k=[1, 2])
        expected = {1: 0.5, 2: 1.0}

        assert actual == expected

    def test_no_values_return_zero(self):
        true_ids = np.array([[1, 2]])
        predicted_ids = np.array(
            [
                [10, 20, 30],
                [40, 50, 50],
            ]
        )

        actual = hit_rate(true_ids=true_ids, predicted_ids=predicted_ids, k=[1, 2])
        expected = {1: 0.0, 2: 0.0}

        assert actual == expected


class TestMeanAveragePrecision:
    @pytest.mark.parametrize(
        "true_ids, predicted_ids, expected",
        [
            (np.array([[1]]), np.array([[1, 2, 3]]), 1.0),
            (np.array([[2]]), np.array([[1, 2, 3]]), 0.5),
            (np.array([[4]]), np.array([[1, 2, 3]]), 0.0),
            (
                    np.array([[1, 2, 3]]),
                    np.array([
                        [1, 2, 3, 4, 5],
                        [1, 2, 10, 11, 12],
                        [6, 7, 8, 9, 10],
                    ]),
                    0.5  # (1+0.5+0) / 3
            ),
        ],
        ids=[
            "Match in first position",
            "Match in second position",
            "No match",
            "Complex matches",
        ]
    )
    def test_simple_examples_return_expected(self, true_ids, predicted_ids, expected):
        actual = mean_average_precision(true_ids=true_ids, predicted_ids=predicted_ids)

        assert np.allclose(actual, expected, atol=1e-5)


class TestMeanAveragePrecisionAtK:
    def test_simple_examples_return_expected(self):
        true_ids = np.array([[1, 2, 3, 4]])
        predicted_ids = np.array([
            [1, 2, 3, 4, 5],
            [1, 2, 10, 11, 12],
            [6, 7, 8, 9, 10],
            [6, 7, 8, 4, 10],
        ])

        k = [1, 2, 5]
        actual = mean_average_precision_at_k(true_ids=true_ids, predicted_ids=predicted_ids, k=k)
        expected = {1: 1 / 4, 2: (1 + 0.5) / 4, 5: (1.0 + 0.5 + 0.25) / 4}

        for current_k in k:
            assert np.allclose(actual[current_k], expected[current_k], atol=1e-5)
