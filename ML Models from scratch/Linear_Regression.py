import numpy as np

class Linear_Regression():
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None

    def fit(self, X_train, y_train):
        # initialize parameters
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X_train, self.weights) + self.bias

            # derivatives
            dw = (1/n_samples) * np.dot(X_train.T, (y_predicted - y_train))
            db = (1/n_samples) * np.sum(y_predicted - y_train)

            # update weights
            self.weights -= self.lr * dw
            # update bias
            self.bias -= self.lr * db

    def predict(self, X_test):
        return np.dot(X_test, self.weights) + self.bias

    def score(X_test, y_test):
        pass

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

class ImprovedLinearRegression:
    def __init__(self, lr=0.01, n_iters=10000, tolerance=1e-6):
        self.lr = lr
        self.n_iters = n_iters
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X_train, y_train):
        # Initialize parameters
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        prev_loss = float('inf')
        
        # Gradient descent
        for i in range(self.n_iters):
            y_predicted = np.dot(X_train, self.weights) + self.bias
            
            # Calculate loss (MSE)
            current_loss = np.mean((y_predicted - y_train) ** 2)
            self.loss_history.append(current_loss)
            
            # Check convergence
            if abs(prev_loss - current_loss) < self.tolerance:
                print(f"Converged after {i} iterations")
                break
                
            prev_loss = current_loss
            
            # Calculate gradients
            dw = (2/n_samples) * np.dot(X_train.T, (y_predicted - y_train))
            db = (2/n_samples) * np.sum(y_predicted - y_train)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Optional: Print progress every 1000 iterations
            if i % 1000 == 0:
                print(f"Iteration {i}, Loss: {current_loss:.4f}")
    
    def predict(self, X_test):
        return np.dot(X_test, self.weights) + self.bias
    
    def score(self, X_test, y_test):
        """Calculate the R² score of the model"""
        y_pred = self.predict(X_test)
        return r2_score(y_test, y_pred)
    
    def get_rmse(self, X_test, y_test):
        """Calculate the Root Mean Squared Error"""
        y_pred = self.predict(X_test)
        return np.sqrt(mean_squared_error(y_test, y_pred))
    
    def plot_loss_history(self):
        """Plot the loss history during training"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.title('Loss History During Training')
        plt.xlabel('Iterations')
        plt.ylabel('Loss (MSE)')
        plt.grid(True)
        plt.show()