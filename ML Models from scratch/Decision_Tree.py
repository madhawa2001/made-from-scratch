from collections import Counter
import numpy as np

def entropy(y):
    """Calculate the entropy of a label array.
    entropy = -sum(p * log2(p)) for each unique label
    where p is the proportion of each label in the array.
    """
    hist = np.bincount(y)
    p = hist / len(y)
    return -np.sum(p[p > 0] * np.log2(p[p > 0]))

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check stopping conditions
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_labels == 1):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Randomly select features if n_features is specified
        feat_indices = np.random.choice(n_features, self.n_features, replace=False)

        # greedily find the best split
        best_feature, best_threshold = self._best_criterion(X, y, feat_indices)
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)
    
    def _best_criterion(self, X, y, feat_indices):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature_idx in feat_indices:
            X_colomn = X[:, feature_idx]
            thresholds = np.unique(X_colomn)
            for threshold in thresholds:
                gain = self._information_gain(y, X_colomn, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold
        
        return split_idx, split_threshold
    
    def _information_gain(self, y, X_colomn, split_threshold):
        """"Calculate the information gain of a split.
            IG = Entropy(parent) - (Weighted Average of Entropy(children))
        """
        #  parent entropy
        parent_entropy = entropy(y)
        # Split the dataset
        left_idxs, right_idxs = self._split(X_colomn, split_threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # Calculate the weighted average of the entropy of the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Information gain
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_colomn, split_threshold):
        """Split the dataset into left and right based on the threshold."""
        left_idxs = np.argwhere(X_colomn <= split_threshold).flatten()
        right_idxs = np.argwhere(X_colomn > split_threshold).flatten()
        return left_idxs, right_idxs

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        """Return the most common label in y."""
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common