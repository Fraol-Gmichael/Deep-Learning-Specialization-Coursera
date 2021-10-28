import utils
import numpy as np
import scipy.optimize as optimize


class LogisticRegression:
    def __init__(self, max_iter=100, lambda_=0):
        self.max_iter = max_iter
        self.lambda_ = lambda_

    def fit(self, X, y):
        theta = self.optimize(X, y, self.lambda_)
        self.w = utils.rank_removal(theta[:-1])
        self.b = theta[-1]      

    def predict(self, X):
        z = np.dot(self.w, X) + self.b
        a = utils.sigmoid(z)
        return (a >= 0.5).astype(int)

    def propagate(self, theta, X, y, lambda_=0):
        
        w = utils.rank_removal(theta[:-1])
        b = theta[-1]

        m = X.shape[1]

        # Forward propagation, calculates cost
        z = np.dot(w, X) + b
        a = utils.sigmoid(z)
        regularize = (lambda_ / (2 * m)) * np.sum(w ** 2)
        cost = (-1/m) * np.sum(y * np.log(a) +
                               (1 - y) * np.log(1 - a)) + regularize

        
        grad_w = (1/m) * np.dot(a - y, X.T).reshape(-1)
        
        grad_b = (1/m) * np.sum(a - y)
        grad_b = np.array([grad_b])
        
        grad = np.hstack([grad_w, grad_b])

        return cost, grad

    def optimize(self, X, y, lambda_=0):
        initial_theta = utils.initialize_theta(X.shape[0])

        costFunction = lambda theta: self.propagate(theta, X, y, lambda_)

        options = {'maxiter': self.max_iter}

        result = optimize.minimize(costFunction, initial_theta, jac=True, method='TNC', options=options)

        return result.x

    def score(self, X, y):
        pred = self.predict(X)
        return np.mean(y==pred)

if __name__ == '__main__':
    X = np.random.rand(20, 100) * 10
    y = np.random.choice([0, 1], size=[1, 100])

    clf = LogisticRegression()
    clf.fit(X, y)