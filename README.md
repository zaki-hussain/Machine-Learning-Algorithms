# Machine Learning Algorithms

## K-Nearest Neighbours Algorithm (KNN)
This algorithm takes three parameters:

1. `k`: The number of neighbours to consider
2. `x_train`: The features of the training data
3. `y_train`: The labels of the training data

The `predict` method takes the parameter `new_data_point` and then calculates the Euclidean distance between that new data point and each of the points in the training data. Then, it sorts these distances in ascending order so that the nearest `k` neighbours can be found. The most frequent label of these nearest neighbours is then returned as a prediction.

I've also used the Iris flower data set, which can be used as sample training data.

## Linear Regression
This creates a least squares regression line. The constructor method takes the parameter `data_points` which should be a two-dimensional NumPy array.

There are three other methods are:
1. `equation` which takes no methods and returns the gradient and y-intercept of the equation
2. `predict` which takes the parameter `x` and plugs it into the equation before returning the corresponding y value
3. `graph` which plots a graph with the training data points and the linear regression line

### Finding the gradient and y-intercept
The residual is the vertical distance between each of the data points and the linear regression line. We want to minimise the residual sum of squares (RSS).

We can write RSS as:
$$
\text{RSS} = \sum _{i=1} ^{n} (y_{i} - (mx_{i} + c))^{2}
$$

Additionally,
$$y = mx + c$$
So it follows that,
$$\bar{y} = m\bar{x} + c
\\ c = \bar{y} - m\bar{x}$$
We can substitute this into RSS and expand the brackets:

$$
\text{RSS} = \sum _{i=1} ^{n} (y_{i} - (mx_{i} + \bar{y} - m\bar{x}))^{2}
\\ = \sum _{i=1} ^{n} (y_{i} - mx_{i} - \bar{y} + m\bar{x})^{2}
\\ = \sum _{i=1} ^{n} ((y_{i} - \bar{y}) - m(x_{i}  - m\bar{x}))^{2}
\\ = \sum _{i=1} ^{n} (y_{i} - \bar{y})^2 -2m(x_{i}  - m\bar{x})(y_{i} - \bar{y}) + m^2(x_{i}  - m\bar{x})^{2}
\\ = S_{yy} -2mS_{xy} + m^2S_{xx}
$$

As we want to minimise RSS, we can differentiate it with respect to $m$ and make it equal to 0 to find which value of m, gives the minimum RSS.
$$
-2S_{xy} + 2mS_{xx} = 0
\\ 2mS_{xx} = 2S_{xy}
\\ mS_{xx} = S_{xy}
\\ m = \frac{S_{xy}}{S_{xx}}
$$

Therefore, we can find $m$, which is our gradient and $c$, which our y-intercept with:
$$
m = \frac{S_{xy}}{S_{xx}}
\\ c = \bar{y} - m\bar{x}
$$