import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, data_points):
        self.x_points = data_points[:, 0]
        self.y_points = data_points[:, 1]
        
        self.x_bar = np.sum(self.x_points) / len(self.x_points)
        self.y_bar = np.sum(self.y_points) / len(self.y_points)

        x_minus_x_bar = self.x_points - self.x_bar

        self.S_xx = np.sum(x_minus_x_bar**2)
        self.S_xy = np.sum((x_minus_x_bar)*(self.y_points - self.y_bar))

        self.m = self.S_xy / self.S_xx
        self.c = self.y_bar - self.m * self.x_bar

    def equation(self):
        return self.m, self.c
    
    def predict(self, x):
        return self.c + self.m * x
    
    def graph(self):
        x_lim = np.max(self.x_points) * 1.1
        y_lim = self.predict(x_lim)
        
        plt.plot([0, x_lim], [self.c, y_lim], color="blue", label = f"y = {self.m:.3g}x + {self.c:.3g}")
        plt.scatter(self.x_points, self.y_points, color="red", label = "Data points")

        plt.xlim(0, x_lim)
        plt.ylim(0, y_lim)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()
