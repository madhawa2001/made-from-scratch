import numpy as np

class BinarySVM():
    def __init__(self, lr=0.001, lamda_param = 0.01, n_iters=1000):
        self.lr = lr
        self.lamda_param = lamda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X_train, y_train):

        # Store the unique classes
        self.classes_ = np.unique(y_train)
        
        # Make sure it's a binary classification problem
        if len(self.classes_) != 2:
            raise ValueError("BinarySVM supports only binary classification")

        y_ = np.where(y_train <= 0, -1, 1)
        n_samples, n_features = X_train.shape

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient descent
        # regularization parameter
        # j = 1/2 * ||w||^2 + lamda * sum(max(0, 1 - y_i * (w^T * x_i + b)))
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X_train):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # dj/dw = 2 * lamda* w
                    # dj/db = 0
                    self.w -= self.lr * (2 * self.lamda_param * self.w)
                else:
                    # dj/dw = 2 * lamda * w - x * y
                    # dj/db = y
                    self.w -= self.lr * (2 * self.lamda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X_test):
        linear_output = np.dot(X_test, self.w) + self.b
        predictions = np.sign(linear_output)
        # Map predictions back to original classes
        return np.where(predictions == -1, self.classes_[0], self.classes_[1])
    

class SVM:
    def __init__(self, lr=0.001, lamda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lamda_param = lamda_param
        self.n_iters = n_iters
        self.models = {}
        self.is_binary = True
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)

        if len(self.classes) == 2:
            # Binary classification case
            self.is_binary = True
            self.models['binary'] = BinarySVM(self.lr, self.lamda_param, self.n_iters)
            self.models['binary'].fit(X, y)
        else:
            # Multi-class: One-vs-Rest strategy
            self.is_binary = False
            for cls in self.classes:
                y_binary = np.where(y == cls, 1, -1)
                model = BinarySVM(self.lr, self.lamda_param, self.n_iters)
                model.fit(X, y_binary)
                self.models[cls] = model

    def predict(self, X):
        if self.is_binary:
            return self.models['binary'].predict(X)
        else:
            # Get scores from each classifier
            scores = []
            for cls in self.classes:
                score = self.models[cls].decision_function(X)
                scores.append(score)

            scores = np.array(scores).T  # shape (n_samples, n_classes)
            predicted_indices = np.argmax(scores, axis=1)
            # Return the original class labels
            return self.classes[predicted_indices]