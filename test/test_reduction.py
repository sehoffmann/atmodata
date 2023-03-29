import sys

import numpy as np
import pytest

from atmodata.reduction import combine_M2, combine_means, Reducer


class TestReducer:
    def test_combine_means(self):
        # Test: Basic
        arr1 = np.arange(4).reshape(2, 2)
        arr2 = np.arange(4).reshape(2, 2) * 10
        mean1 = combine_means(1, arr1, 1, arr2)
        np.testing.assert_array_almost_equal(mean1, np.array([[0.0, 5.5], [11.0, 16.5]]))

        # Test: variyng number of samples
        mean1 = combine_means(10, arr1, 2, arr2)
        np.testing.assert_array_almost_equal(mean1, np.array([[0.0, 2.5], [5.0, 7.5]]))

        # Test: numeric stability
        mean = np.asarray(0.0, dtype=np.float32)
        faces = [arr for arr in np.arange(6, dtype=np.float32) + np.float32(1)]
        for n in range(int(6e3)):
            mean = combine_means(np.float32(n), mean, np.float32(1), faces[n % 6])
        np.testing.assert_array_almost_equal(mean, 3.5, decimal=5)

    def test_combine_M2(self):
        # Test: [-1,-1,-1,-1,-1,1,1,1,1,1] combined with [0,6]
        n_a = 10
        mean_a = 0.0
        M2_a = 10 * 1.0
        n_b = 2
        mean_b = 3.0
        M2_b = 2 * 9.0
        M2 = combine_M2(n_a, mean_a, M2_a, n_b, mean_b, M2_b)
        np.testing.assert_array_almost_equal(M2, 43)

        # Test: [2,3] combinded with [20, 30]
        n_a = 2
        mean_a = 2.5
        M2_a = 0.5
        n_b = 2
        mean_b = 25.0
        M2_b = 50.0
        M2 = combine_M2(n_a, mean_a, M2_a, n_b, mean_b, M2_b)
        np.testing.assert_array_almost_equal(M2, 556.75)

        # Test: procedural test
        for idx in [900, 700, 500]:
            base = np.arange(1000)
            arr1 = base[:idx]
            arr2 = base[idx:]
            M2_a = np.sum((arr1 - arr1.mean()) ** 2)
            M2_b = np.sum((arr2 - arr2.mean()) ** 2)
            M2 = combine_M2(len(arr1), arr1.mean(), M2_a, len(arr2), arr2.mean(), M2_b)
            expected = np.sum((base - base.mean()) ** 2)
            np.testing.assert_array_almost_equal(M2, expected)

    def test_Reducer(self):
        arr = np.arange(4).reshape(2, 2)
        reducer = Reducer(axis=1)

        # Test: First reducer
        reducer.update(arr)
        np.testing.assert_array_almost_equal(reducer.mean, np.array([0.5, 2.5]))
        np.testing.assert_array_almost_equal(reducer.std, np.array([0.5, 0.5]))
        np.testing.assert_array_almost_equal(reducer.min, np.array([0, 2]))
        np.testing.assert_array_almost_equal(reducer.max, np.array([1, 3]))

        # Test: incremental reducer
        reducer.update(arr * 10)
        np.testing.assert_array_almost_equal(reducer.mean, np.array([2.75, 13.75]))
        np.testing.assert_array_almost_equal(reducer.std, np.array([4.205651, 11.797775]))
        np.testing.assert_array_almost_equal(reducer.min, np.array([0, 2]))
        np.testing.assert_array_almost_equal(reducer.max, np.array([10, 30]))

        # Test: PCA
        reducer = Reducer(axis=1, pca=True)
        # fmt: off
        arr = np.array([[0, 0],
                        [0, 2]])
        # fmt: on
        for _ in range(20):
            reducer.update(arr)

        np.testing.assert_array_almost_equal(
            reducer.mean, np.array([0.0, 1.0])
        )  # make sure means and variances are still correctly calculated
        np.testing.assert_array_almost_equal(reducer.std, np.array([0.0, 1.0]))
        np.testing.assert_array_almost_equal(
            reducer.pca_explained_variance_ratio, np.array([1.0, 0.0])
        )  # theres only a single axis of variation

        failed = False
        try:
            np.testing.assert_array_almost_equal(reducer.pca_components[0], np.array([0.0, 1.0]))
        except AssertionError:
            failed = True

        if failed:
            np.testing.assert_array_almost_equal(reducer.pca_components[0], np.array([0.0, -1.0]))

        # TODO: Test axis=None
        # TODO: Test axis=list


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
