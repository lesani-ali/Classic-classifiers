import numpy as np


class KNN:

    # Constructor
    def __init__(self, k=5, _type="classification"):
        """
        Initialize k-Nearest Neighbors (KNN) with specified parameters.

        :param k: int, number of neighbors to use
        :param _type: str, either 'classification' or 'regression'
        """
        self.k = k
        self._type = _type  # 'classification' or 'regression'

    def fit(self, X_train, y_train):
        """
        Fit the KNN model to the training data.

        :param X_train: numpy array, training data
        :param y_train: numpy array, training labels
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, samples):
        """
        Predict the class labels for the input samples.

        :param samples: numpy array, input samples
        :return: numpy array, predicted labels
        """
        y_predicted = np.array([self._predict(sample.reshape(-1, 1)) for sample in samples])
        return y_predicted

    def _predict(self, x):
        """
        Predict the class label for a single sample.

        :param x: numpy array, input sample
        :return: predicted label
        """
        distances = [np.linalg.norm(x - sample.reshape(-1, 1)) for sample in self.X_train]
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = self.y_train[k_indices]

        if self._type == "classification":
            prediction = max(set(k_nearest_labels.tolist()), key=k_nearest_labels.tolist().count)
        elif self._type == "regression":
            prediction = np.mean(k_nearest_labels)
        else:
            raise ValueError(
                "Invalid type specified. Use 'classification' or 'regression'."
            )

        return prediction
