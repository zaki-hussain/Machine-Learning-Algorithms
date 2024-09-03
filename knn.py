import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k
    
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, new_data_point):
        self.new_data_point = new_data_point
        self.distances = np.sqrt(np.sum((self.x_train-self.new_data_point)**2, axis=1))
        self.ordered_indices = np.argsort(self.distances)