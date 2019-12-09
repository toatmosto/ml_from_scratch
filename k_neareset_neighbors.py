import numpy as np
import heapq
from collections import Counter

class KNN(object):
    def __init__(self, num_neighbors: int):
        self.num_neighbors = num_neighbors
        self.x_train = None
        self.y_train = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, data: np.ndarray) -> np.ndarray:
        """

        :param data: n-d numpy array, required
            each row is a data record
        :return:
        """
        if len(data.shape) == 1:
            data = data.reshape([1, data.shape[0]])
        pred = []
        for i, x in enumerate(data):
            # calculate distances b/w current obs and all obs in x_train
            dists = np.sqrt(np.sum((self.x_train - x)**2, axis=1))
            # create a min heap with its index
            dist_heap = [(dist, idx) for idx, dist in enumerate(dists)]
            heapq.heapify(dist_heap)
            # get the top
            y_neighbors = []
            for j in range(self.num_neighbors):
                j_th_closest_idx = heapq.heappop(dist_heap)[1]
                y_neighbors.append(self.y_train[j_th_closest_idx])
            y_count_i = Counter(y_neighbors)
            pred.append(y_count_i.most_common(1)[0][0])

        return np.array(pred)
