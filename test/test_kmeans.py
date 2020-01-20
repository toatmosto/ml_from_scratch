from kmeans import KMeans
import numpy as np


def test_update_cluster_center():
    test_model = KMeans(n_clust=2, random_seed=0)
    test_x_train = np.array([[1, 2],
                             [1, 4],
                             [1, 0],
                             [10, 2],
                             [10, 4],
                             [10, 0]])
    test_model.classes = np.array([1, 0, 1, 0, 1, 0])
    test_model._update_cluster_center(test_x_train)
    expected_centers = np.array([[7., 2.], [4., 2.]])
    np.testing.assert_allclose(test_model.centers, expected_centers,
                               rtol=0, atol=1e-7)


def test_update_membership():
    test_model = KMeans(n_clust=2, random_seed=0)
    test_x_train = np.array([[1, 2],
                             [1, 4],
                             [1, 0],
                             [10, 2],
                             [10, 4],
                             [10, 0]])
    test_model.classes = np.array([1, 0, 1, 0, 1, 0])
    test_model.centers = np.array([[7., 2.], [4., 2.]])
    test_model._update_membership(test_x_train)
    expected_classes = np.array([1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(test_model.classes, expected_classes)


def test_predict():
    test_model = KMeans(n_clust=2, random_seed=0)
    x_test = np.array([[1, 2], [30, 2]])
    test_model.centers = np.array([[7., 2.], [4., 2.]])
    expected_classes = np.array([1, 0])
    np.testing.assert_array_equal(test_model.predict(x_test), expected_classes)


def test_fit():
    test_model = KMeans(n_clust=2, random_seed=100)
    test_x_train = np.array([[1, 2],
                             [1, 4.1],
                             [1, 0],
                             [10, 2.1],
                             [10, 4.1],
                             [10, 0]])
    test_model.fit(test_x_train)
    expected_classes = np.array([0, 0, 0, 1, 1, 1])
    expected_centers = np.array([[1., 2.0333333], [10., 2.0666667]])
    np.testing.assert_array_equal(test_model.classes, expected_classes)
    np.testing.assert_allclose(test_model.centers, expected_centers)
