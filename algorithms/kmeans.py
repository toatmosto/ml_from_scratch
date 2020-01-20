import numpy as np


class KMeans(object):
    def __init__(self, n_clust: int, random_seed: int=None):
        """
        :param n_clust: number of clusters, cluster id starts at 0
        """
        self.n_clust = n_clust
        self.random_seed = random_seed
        # each row of self.centers is a center, so it has n_clust rows
        self.centers: np.ndarray = None
        self.classes: np.ndarray = None

    def fit(self, x_train: np.ndarray):
        """
        execute the k means clustering algorithm using data given in x_train.
        step 1: initialize cluster membership for each obs
        step 2: update cluster center ->  ->
        step 3: create new membership to the closest cluster center
        step 4: check if new membership is same as the old one, if yes, stop,
            otherwise go to step 2
        :param x_train:
        :return:

        Problem for current version:
        1. sometimes the centers generated are the same, which will lead to all obs go to one clusters
            e.g. set random seed = 10 in test_fit()
        2. local minima
        """
        n_obs: int = x_train.shape[0]
        # step 1
        self._init_cluster_center(n_obs)

        # starts iterative re-allocation
        num_updates = 1
        while num_updates > 0:
            # step 2
            self._update_cluster_center(x_train)

            # step 3
            num_updates = self._update_membership(x_train)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        predict the cluster membership for the new data using the fitted cluster centers
        :return:
        """
        # create distance matrix,
        # dist_mat[i][j] is the distance b/w obs i and center j
        # a more memory efficient way is to calculate distances for each obs
        # one at a time, but it will be slower in python
        dist_mat = np.ndarray([x_test.shape[0], self.n_clust])
        for j in range(self.n_clust):
            dist_mat[:, j] = np.sqrt(np.sum((x_test - self.centers[j]) ** 2, axis=1))
        return np.argmin(dist_mat, axis=1)

    def _init_cluster_center(self, n_obs: int):
        """
        initialize cluster membership, cluster id starts from 0
        :param n_obs:
        :return:
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.classes = np.random.choice(self.n_clust, n_obs)

    def _update_cluster_center(self, x_train: np.ndarray):
        """
        update the cluster centers using the current cluster membership
        in self.classes
        :param x_train:
        :return:
        """
        if self.centers is None:
            self.centers = np.ndarray([self.n_clust, x_train.shape[1]])
        # a more memory efficient and faster way is to go through every obs
        # in x_train and add to the right center, but this may be slower in
        # python (though I am not sure), as np.sum is probably using C in the back
        for i in range(self.n_clust):
            x_idx_cluster_i = np.where(self.classes == i)[0]
            x_cluster_i = x_train[x_idx_cluster_i, :]
            # sum along dimension 0 (row), meaning dimension 0 will be collapsed
            self.centers[i] = np.mean(x_cluster_i, axis=0)

    def _update_membership(self, x_train: np.ndarray) -> int:
        """
        update the membership self.classes, using the current self.centers
        and distance between each x_train to each center
        :param x_train:
        :return:
        """
        new_memberships = self.predict(x_train)
        num_updates = int(np.sum(self.classes != new_memberships))
        self.classes = new_memberships
        return num_updates
