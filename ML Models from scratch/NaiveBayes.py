import numpy as np

class NaiveBayes:
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.classes = np.unique(y_train)
        n_classes = len(self.classes)
    
        # initialize mean, variance and prior
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.variance = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for c in self.classes:
            X_c = X_train[y_train == c]
            self.mean[c,:] = X_c.mean(axis=0)
            self.variance[c,:] = X_c.var(axis=0)
            self.priors[c] = X_c.shape[0] / n_samples

    def predict(self, X_test):
        y_pred = [self._predict(x) for x in X_test]
        return y_pred
    
    def _predict(self, x):
        posteriors = []
        
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]
    
    # probablity density function
    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        variance = self.variance[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator