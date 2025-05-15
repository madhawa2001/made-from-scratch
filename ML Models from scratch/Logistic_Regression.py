import numpy as np

class Logistic_Regression():
    def __init__(self, lr=0.001, n_iters=1000, tolerance=1e-6):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.tolerance = tolerance
        self.loss_history = []

    def fit(self, X_train, y_train):
        # initialize parameters
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        prev_loss = float('inf')

        # gradient descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X_train, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Calculate loss (MSE)
            current_loss = np.mean((y_predicted - y_train) ** 2)
            self.loss_history.append(current_loss)

            # Check for convergence
            if abs(prev_loss - current_loss) < self.tolerance:
                print(f"Converged after {i} iterations")
                break

            prev_loss = current_loss

            # derivatives
            dw = (2/n_samples) * np.dot(X_train.T, (y_predicted - y_train))
            db = (2/n_samples) * np.sum(y_predicted - y_train)

            # update weights
            self.weights -= self.lr * dw
            # update bias
            self.bias -= self.lr * db

    def predict(self, X_test):
        linear_model = np.dot(X_test, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_class)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))