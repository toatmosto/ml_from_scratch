import numpy as np


class LogisticRegression:
    def __init__(self):
        self.parameters: np.ndarray = None
        self.loss: np.float = None
        self.loss_history: list = []
        self.parameters_history: list = []

    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            tol=0.000001, lr=0.01, max_iter = 100):
        """
        a method that takes in the training data and training labels,
        assuming the data is clean without missing values, and we assume the
        intercept is always there.
        :param x_train:
        :param y_train:
        :return:

        TODO implement other optim_method, e.g. Newton-raphson, or Gauss-Newton
        check https://paper.dropbox.com/doc/Yuans-implementation-of-ML-algorithms-from-scrach--
        AtOQkoFoTlWAgwiaYC4ytLhKAQ-kfkoIrtX2JQYk2r2EXGS9
        for details of Newton's method in logistic regression
        """
        prev_params = np.zeros((x_train.shape[1],)) + 1.0
        curr_params = np.zeros((x_train.shape[1],))
        while np.sum(np.abs(curr_params - prev_params)) > tol and \
                len(self.loss_history) <= max_iter:
            # store current loss and parameters
            self.parameters_history.append(curr_params)
            self.loss_history.append(self._loss(x_train, y_train, curr_params))
            # calculate the gradient
            grad = self._gradient(x_train, y_train, curr_params)
            # update parameters
            prev_params = curr_params
            curr_params = curr_params - lr * grad

        self.parameters = curr_params
        self.loss = self._loss(x_train, y_train, curr_params)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        predict the probability for the given input x
        :param x:
        :return:
        """
        return self._predict_prob(x, self.parameters)

    def _predict_prob(self, x: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """
        :param x: numpy array of shape (n, d)
        :param beta: numpy array of shape (d,) or (d, 1)
        :return: numpy array of shape (n,)
        """
        eita = np.matmul(x, beta)
        return self._inverse_logit(eita)

    @staticmethod
    def _inverse_logit(x: np.ndarray) -> np.ndarray:
        """
        :param x: a numpy array of any shape
        :return: a numpy array of same shape as x
        """
        return 1.0 / (1.0 + np.exp(-x))

    def _loss(self, x_train: np.ndarray, y_train: np.ndarray, beta: np.ndarray) -> np.float:
        """
        :param x_train: numpy array of shape (n, d)
        :param y_train: numpy array of shape (n,) or (n, 1)
        :param beta: numpy array of shape (d, ) or (d, 1)
        :return: numpy float64
        """
        # TODO check input validity, size, missing, etc.
        # X * beta, should be a vector
        eita = np.matmul(x_train, beta)
        # loss or negative log likelihood
        log_likelihood = -y_train * np.log(1.0 + np.exp(-eita)) + \
            (1.0 - y_train) * np.log(1.0 + np.exp(eita))
        return -np.sum(log_likelihood)

    def _gradient(self, x_train: np.ndarray, y_train: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """

        :param x_train: numpy array of shape (n, d)
        :param y_train: numpy array of shape (n,) or (n, 1)
        :param beta: numpy array of shape (d, ) or (d, 1)
        :return: numpy array of shape (d,)
        """
        # temp should have shape n by 1
        temp = self._inverse_logit(np.matmul(x_train, beta)) - y_train
        return np.matmul(x_train.T, temp)

