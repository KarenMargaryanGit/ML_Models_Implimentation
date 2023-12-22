import numpy as np


class KNearestNeighbor:
    """ a kNN classifier with hamming distance """

    def __init__(self, X_train, y_train):
        """
    Initializing the KNN object

    Inputs:
    - X_train: A numpy array or pandas DataFrame of shape (num_train, D) 
    - y_train: A numpy array or pandas Series of shape (N,) containing the training labels
    """
        self.X_train = X_train
        self.y_train = y_train

    def fit_predict(self, X_test, k=1):
        """
    This method fits the data and predicts the labels for the given test data.
    For k-nearest neighbors fitting (training) is just
    memorizing the training data.
    Inputs:
    - X_test: A numpy array or pandas DataFrame of shape (num_test, D)
    - k: The number of nearest neighbors.

    Returns:
    - y: A numpy array or pandas Series of shape (num_test,) containing predicted labels
    """
        dists = self.compute_distances(X_test)
        return self.predict_labels(dists, k=k)

    def compute_distances(self, X_test):
        """
    Compute the hamming distance between each test point in X_test and each training point
    in self.X_train.

    Inputs:
    - X_test: A numpy array or pandas DataFrame of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the hamming distance between the ith test point and the jth training
      point.
    """
        dists = np.zeros((X_test.shape[0], self.X_train.shape[0]))

        for i in range(X_test.shape[0]):
            # print(X_test.iloc[i] != self.X_train)
            dists[i,:] = np.sum(self.X_train != X_test.iloc[i,:], axis=1)
            
            
        # print(dists)        
        return dists

    def predict_labels(self, dists, k=1):
        """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance between the ith test point and the jth training point.

    Returns:
    - y: A numpy array or pandas Series of shape (num_test,) containing the
    predicted labels for the test data
    """
        y_pred = []

        for i in range(dists.shape[0]):
            k_closest = np.argsort(dists[i])[:k]
            y_labels = self.y_train.iloc[k_closest]            
            # print(y_labels.value_counts().index[0])
            y_pred.append(y_labels.value_counts().index[0])

        return np.array(y_pred)