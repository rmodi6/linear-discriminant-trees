import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class Utils:
    @staticmethod
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum(np.square(x1 - x2)))


class TreeNode:
    def __init__(self, val=None, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None


class LabelDetails:
    def __init__(self, X, y, label):
        self.X = X
        self.y = y
        self.label = label
        self.sample_count = X.shape[0]
        self.mean = self.X.describe().loc['mean'].values.reshape(-1, 1)

    def get_values(self):
        return self.X.values

    def distance(self, other):
        return Utils.euclidean_distance(self.mean, other.mean)

    def __eq__(self, other):
        return isinstance(other, LabelDetails) and self.label == other.label and np.all(self.mean == other.mean)

    def __ne__(self, other):
        return not self.__eq__(other)


class LDTree:
    def __init__(self):
        self.root = None

    def fit(self, X, y):
        label_details = self.create_labels(X, y)
        self.root = self.build_tree(label_details)

    def predict(self, X_test):
        y_preds = []
        for i, row in X_test.iterrows():
            y_preds.append(self.classify(row.values.reshape(-1, X_test.shape[1]), self.root))
        return y_preds

    def classify(self, X, node):
        if node.is_leaf():
            return node.val
        if node.val.predict(X) == 0:
            return self.classify(X, node.left)
        if node.val.predict(X) == 1:
            return self.classify(X, node.right)

    @staticmethod
    def build_linear_discriminant(left_split, right_split):
        features, labels = None, None

        for label_detail in left_split:
            feature_values = label_detail.get_values()
            if features is None:
                features = feature_values
                labels = np.zeros(features.shape[0])
            else:
                features = np.append(features, feature_values, axis=0)
                labels = np.append(labels, np.zeros(feature_values.shape[0]))

        for label_detail in right_split:
            feature_values = label_detail.get_values()
            if features is None:
                features = feature_values
                labels = np.ones(features.shape[0])
            else:
                features = np.append(features, feature_values, axis=0)
                labels = np.append(labels, np.ones(feature_values.shape[0]))

        return LDA().fit(features, labels)

    def build_tree(self, label_details):
        if len(label_details) == 1:
            # create leaf node
            leaf_node = TreeNode(label_details[0].label)
            return leaf_node

        left_split, right_split = self.heuristic_split(label_details)
        left_split, right_split = self.exchange(left_split, right_split, label_details)

        left_node, right_node = self.build_tree(left_split), self.build_tree(right_split)
        lda_model = self.build_linear_discriminant(left_split, right_split)

        return TreeNode(lda_model, left_node, right_node)

    @staticmethod
    def heuristic_split(label_details):
        maximum_distance = float('-inf')
        splits = {}
        for i in range(len(label_details)):
            for j in range(i, len(label_details)):
                distance = label_details[i].distance(label_details[j])
                if distance > maximum_distance:
                    maximum_distance = distance
                    splits = {'left': [label_details[i]], 'right': [label_details[j]]}
        maximum_distance_splits = splits['left'] + splits['right']
        for label_detail in label_details:
            if label_detail not in maximum_distance_splits:
                left_distance = label_detail.distance(splits['left'][0])
                right_distance = label_detail.distance(splits['right'][0])
                if left_distance < right_distance:
                    splits['left'].append(label_detail)
                else:
                    splits['right'].append(label_detail)

        return splits['left'], splits['right']

    def exchange(self, left_split, right_split, label_details):
        maximum_information_gain = self.compute_information_gain(left_split, right_split)
        best_partition = (left_split, right_split)
        for label_detail in label_details:
            left_split_copy = left_split.copy()
            right_split_copy = right_split.copy()

            if label_detail in left_split:
                left_split_copy = [o for o in left_split_copy if o != label_detail]
                right_split_copy.append(label_detail)
            else:
                right_split_copy = [o for o in right_split_copy if o != label_detail]
                left_split_copy.append(label_detail)

            if len(left_split_copy) == 0 or len(right_split_copy) == 0:
                continue

            information_gain = self.compute_information_gain(left_split_copy, right_split_copy)

            if information_gain > maximum_information_gain:
                maximum_information_gain = information_gain
                best_partition = (left_split_copy, right_split_copy)

        return best_partition

    def compute_information_gain(self, left_split, right_split):
        if len(left_split) == 0 or len(right_split) == 0:
            return float('-inf')

        lda_model = self.build_linear_discriminant(left_split, right_split)

        total_left_samples = sum([ld.sample_count for ld in left_split])
        total_right_samples = sum([rd.sample_count for rd in right_split])
        total = total_left_samples + total_right_samples

        e0 = 0.0
        for detail in left_split + right_split:
            e0 = e0 + self.compute_entropy(detail.sample_count, total)

        left_predictions = []
        right_predictions = []

        for split in [left_split, right_split]:
            l, r = self.get_lda_predictions(split, lda_model)
            left_predictions = left_predictions + l
            right_predictions = right_predictions + r

        information_gain = e0
        for predictions, total_count in [(left_predictions, total_left_samples),
                                         (right_predictions, total_right_samples)]:
            val = 0.0
            for prediction in predictions:
                if prediction != 0.0:
                    temp = float(prediction) / total_count
                    val = val + temp * np.log2(temp)
            information_gain += val * total_count / total

        return information_gain

    @staticmethod
    def compute_entropy(prediction, total):
        val = float(prediction) / total
        return -1.0 * val * np.log2(val)

    @staticmethod
    def get_lda_predictions(split, lda):
        left_predictions = []
        right_predictions = []
        for label_detail in split:
            left_count = 0
            right_count = 0
            rows = label_detail.X.shape[0]
            X = label_detail.X.values
            for i in range(rows):
                if lda.predict([X[i]]) == 0:
                    left_count += 1
                else:
                    right_count += 1

            left_predictions.append(left_count)
            right_predictions.append(right_count)

        return left_predictions, right_predictions

    @staticmethod
    def create_labels(X, y):
        unique_classes = set(y.unique())
        label_details = []
        for class_type in unique_classes:
            yi = y[y == class_type]
            indexes = yi.index
            xi = X.loc[indexes]
            label_detail = LabelDetails(xi, yi, class_type)

            label_details.append(label_detail)

        return label_details


if __name__ == '__main__':
    # Iris Dataset
    dataset = pd.read_csv('data/iris.data')
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    for test_size in [0.2, 0.25, 0.3]:
        training_accuracies, validation_accuracies = [], []
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

            model = LDTree()
            model.fit(X_train, y_train)
            training_accuracies.append(accuracy_score(y_train, model.predict(X_train)))
            validation_accuracies.append(accuracy_score(y_test, model.predict(X_test)))

        print(test_size, np.mean(training_accuracies), np.mean(validation_accuracies))

    # Breast Cancer Dataset
    dataset = pd.read_csv('data/breast_cancer.data')
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    for test_size in [0.2, 0.25, 0.3]:
        training_accuracies, validation_accuracies = [], []
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

            model = LDTree()
            model.fit(X_train, y_train)
            training_accuracies.append(accuracy_score(y_train, model.predict(X_train)))
            validation_accuracies.append(accuracy_score(y_test, model.predict(X_test)))

        print(test_size, np.mean(training_accuracies), np.mean(validation_accuracies))

    # Ecoli Dataset
    dataset = pd.read_csv('data/ecoli.data', sep='\s+')
    X = dataset.iloc[:, 1:-1]
    y = dataset.iloc[:, -1]

    for test_size in [0.2, 0.25, 0.3]:
        training_accuracies, validation_accuracies = [], []
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

            model = LDTree()
            model.fit(X_train, y_train)
            training_accuracies.append(accuracy_score(y_train, model.predict(X_train)))
            validation_accuracies.append(accuracy_score(y_test, model.predict(X_test)))

        print(test_size, np.mean(training_accuracies), np.mean(validation_accuracies))
