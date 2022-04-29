import numpy as np
import h5py


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(int)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


class DNN():
    def __init__(self, max_iter=3000, learning_rate=0.01, layers=[10, 1], lambda_=0):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.layers = layers
        self.lambda_ = lambda_
        self.num_layers = len(self.layers)

    def initialize_parameters(self, multi=False):
        W = []
        B = []
        n_right = self.X.shape[0]
        np.random.seed(1)
        for n_left in self.layers:
            if multi:
                w = np.random.randn(n_left, n_right) / np.sqrt(n_right)
            else:
                w = np.random.randn(n_left, n_right) * 0.01

            b = np.zeros((n_left, 1))

            n_right = n_left

            W.append(w)
            B.append(b)

        self.W, self.B = np.array(W, dtype=object), np.array(B, dtype=object)

    def fit(self, X, y):
        self.X = X
        self.y = y

        if self.num_layers > 2:
            self.initialize_parameters(multi=True)
        else:
            self.initialize_parameters(multi=False)

        for each in range(self.max_iter):
            A, Z = self.forward_prop(self.X)
            dW, dB = self.backward_prop(A, Z)
            self.update_parameters(dW, dB)
            cost = self.compute_cost(X, y)
            if ((each+1) % 100 == 0):
                print(f"cost at iteration {each+1} = {cost}")
        print(f'Final cost after {self.max_iter} is {cost}')

    def forward_prop(self, X):
        A = [X]
        Z = []

        for idx in range(self.num_layers - 1):
            z = np.dot(self.W[idx], A[idx]) + self.B[idx]
            a = relu(z)
            A.append(a)
            Z.append(z)

        z = np.dot(self.W[idx+1], A[idx+1]) + self.B[idx+1]
        a = sigmoid(z)
        A.append(a)
        Z.append(z)

        return A, Z

    def compute_cost(self, X, y):
        m = y.shape[1]
        y_hat = self.forward_prop(X)[0][self.num_layers]
                
        regularize = np.sum(np.array([np.sum(w ** 2) for w in self.W]))

        cost = -(1/m) * np.sum((y * np.log(y_hat)) +
                               ((1 - y) * np.log(1-y_hat))) + (self.lambda_ / (2 * m)) * regularize

        return cost

    def backward_prop(self, A, Z):
        def dW_and_dB(dz, a, w):
            dw = (1/m) * np.dot(dz, a.T) + (self.lambda_/m) * w
            db = (1/m) * np.sum(dz, axis=1, keepdims=True)

            return dw, db

        m = self.y.shape[1]
        dZ = [A[self.num_layers] - self.y]
        dw, db = dW_and_dB(dZ[0], A[self.num_layers-1], self.W[self.num_layers-1])
        dW = [dw]
        dB = [db]

        for idx in reversed(range(1, self.num_layers)):
            dz = np.dot(self.W[idx].T, dZ[-1]) * relu_derivative(Z[idx-1])
            dZ.append(dz)
            dw, db = dW_and_dB(dz, A[idx - 1], self.W[idx-1])
            dW.append(dw)
            dB.append(db)

        dW, dB = np.flip(np.array(dW, dtype=object)), np.flip(
            np.array(dB, dtype=object))

        return dW, dB

    def update_parameters(self, dW, dB):
        self.W -= self.learning_rate * dW
        self.B -= self.learning_rate * dB

    def predict(self, X):
        y_hat = self.forward_prop(X)[0][self.num_layers]
        y_hat = (y_hat > 0.5).astype(int)

        return y_hat

    def score(self, X, y):
        y_hat = self.predict(X)
        sc = np.mean(y_hat == y)

        return sc


def flatme(me, standard=True):
    if standard:
        result = (me.reshape(me.shape[0], -1)/255).T
    else:
        result = (me.reshape(me.shape[0], -1)).T
    return result


def prepare_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")

    keys = list(train_dataset.keys())
    list_classes, train_set_x, train_set_y = keys
    X_train = np.array(train_dataset[train_set_x])
    y_train = np.array(train_dataset[train_set_y])

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    keys = list(test_dataset.keys())
    list_classes, test_set_x, test_set_y = keys

    X_test = np.array(test_dataset[test_set_x])
    y_test = np.array(test_dataset[test_set_y])
    X_test, y_test = flatme(X_test), flatme(y_test, False)
    X_train, y_train = flatme(X_train), flatme(y_train, False)

    return X_train, y_train, X_test, y_test, list_classes


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, classes = prepare_dataset()
    clf = DNN(max_iter=2500, learning_rate=0.0075, layers=[20, 7, 5, 1], lambda_=0.3)
    clf.fit(X_train, y_train)

    score = clf.score(X_train, y_train)
    print("Score:", score)
    score = clf.score(X_test, y_test)
    print("Score:", score)
