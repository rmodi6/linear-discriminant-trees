import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class Utils:
    """
    Utility class to store static util methods
    """
    @staticmethod
    def euclidean_distance(x1, x2):
        """
        Function to calculate euclidean distance between two vectors
        :param x1: vector 1
        :param x2: vector 2
        :return: euclidean distance between x1 and x2
        """
        return np.sqrt(np.sum(np.square(x1 - x2)))


class TreeNode:
    """
    A node in the decision tree. Can be an intermediate node or a leaf node
    """
    def __init__(self, val=None, left=None, right=None):
        """
        Init method
        :param val: value at node
        :param left: left child of node
        :param right: right child of node
        """
        self.val = val
        self.left = left
        self.right = right

    def is_leaf(self):
        """
        Method to test if the node is a leaf node
        :return: true if it is a leaf node
        """
        return self.left is None and self.right is None


class LabelDetails:
    """
    Class to store the details of data for splitting at each node
    """
    def __init__(self, X, y, label):
        """
        Init method
        :param X: features
        :param y: labels
        :param label: label name
        """
        self.X = X
        self.y = y
        self.label = label
        self.sample_count = X.shape[0]
        self.mean = self.X.describe().loc['mean'].values.reshape(-1, 1)

    def get_values(self):
        """
        Method to return a numpy array of features
        :return: ndarray of X
        """
        return self.X.values

    def distance(self, other):
        """
        Method to compute the distance between the means of current object and other object
        :param other: other labeldetail object with which distance is to be calculated
        :return: distance between their means
        """
        return Utils.euclidean_distance(self.mean, other.mean)

    def __eq__(self, other):
        """
        Overrides default implementation to check equality between two label details objects
        :param other: other labeldetail object
        :return: true if the objects are equal
        """
        return isinstance(other, LabelDetails) and self.label == other.label and np.all(self.mean == other.mean)

    def __ne__(self, other):
        """
        Overrides default implementation to check inequality between two label details objects
        :param other: other labeldetail object
        :return: true if the objects are not equal
        """
        return not self.__eq__(other)


class LDTree:
    """
    Class to implement the Linear Discriminant Tree classifier. The class structure is very similar to the
    DecisionTreeClassifier of sklearn
    """
    def __init__(self):
        """
        Init method
        """
        self.root = None

    def fit(self, X, y):
        """
        Method to train the LD Tree based on input features and thier labels
        :param X: training features
        :param y: training labels
        :return:
        """
        # Create the label_details object which can be used for deciding split of classes
        label_details = self.create_labels(X, y)
        # Build the tree and assign the root node
        self.root = self.build_tree(label_details)
        return self

    def predict(self, X_test):
        """
        Method to predict labels for test data
        :param X_test: pandas dataframe of test features
        :return: list of predicted class labels for X_test
        """
        # Raise exception is model is not trained
        if self.root is None:
            raise Exception("Train the model first using training data")
        y_preds = []
        for i, row in X_test.iterrows():  # For each row in test data
            # Make a recursive call to classify() method to get the prediction using the trained decision tree
            y_preds.append(self.classify(row.values.reshape(-1, X_test.shape[1]), self.root))
        return y_preds

    def classify(self, X, node):
        """
        Recursive method to classify a data point
        :param X: data point (features)
        :param node: current node in the tree
        :return: predicted class label for X
        """
        # Return the node label if it is a leaf node
        if node.is_leaf():
            return node.val
        # Based on lda prediction at current node recursively traverse left or right child
        return self.classify(X, node.left) if node.val.predict(X) == 0 else self.classify(X, node.right)

    @staticmethod
    def create_labels(X, y):
        """
        Method to create label details for the training data
        :param X: features
        :param y: labels
        :return: list of label details for each class type
        """
        unique_classes = set(y.unique())
        label_details = []
        for class_type in unique_classes:
            yi = y[y == class_type]
            indexes = yi.index
            xi = X.loc[indexes]
            label_detail = LabelDetails(xi, yi, class_type)
            label_details.append(label_detail)

        return label_details

    def build_tree(self, label_details):
        if len(label_details) == 1:
            # create leaf node with value as label of remaining class and no child nodes
            leaf_node = TreeNode(label_details[0].label)
            return leaf_node

        # Compute the initial split based on the heuristic
        left_split, right_split = self.heuristic_split(label_details)
        # Optimize the splits using exchange method as described in the paper
        left_split, right_split = self.exchange(left_split, right_split, label_details)

        # Create the LDA model at the current node using the left and right splits
        lda_model = self.build_linear_discriminant(left_split, right_split)
        # Recursively build the subtree for the left and right child
        left_node, right_node = self.build_tree(left_split), self.build_tree(right_split)

        # Return a TreeNode with left and right child and lda_model as the value
        return TreeNode(lda_model, left_node, right_node)

    @staticmethod
    def heuristic_split(label_details):
        """
        Method to create an initial split at a node based on the heuristic mentioned in the paper
        :param label_details: list of label details at the current node
        :return: maximum distance apart left and right splits of label details
        """
        maximum_distance = float('-inf')
        splits = {}
        # Compute the initial label details for left and right split that are maximum distance apart
        for i in range(len(label_details)):
            for j in range(i, len(label_details)):
                distance = label_details[i].distance(label_details[j])
                if distance > maximum_distance:
                    maximum_distance = distance
                    splits = {'left': [label_details[i]], 'right': [label_details[j]]}
        maximum_distance_splits = splits['left'] + splits['right']
        # Assign label detail to left or right split whichever is closer to it based on the initial split label details
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
        """
        Exchange method implementation for optimizing the splits at current node
        :param left_split: list of label details in the initial left split
        :param right_split: list of label details in the initial right split
        :param label_details: list of all the label details
        :return:
        """
        # Compute the current information gain for the initial splits
        maximum_information_gain = self.compute_information_gain(left_split, right_split)
        best_partition = (left_split, right_split)
        # For each label detail exchange it i.e. if the label detail is in right split move it to the left split
        # and if the label detail is in left split move it to the right split and compute the information gain
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

            # Update the best partition if current information gain is higher than maximum information gain until now
            if information_gain > maximum_information_gain:
                maximum_information_gain = information_gain
                best_partition = (left_split_copy, right_split_copy)

        # Return the best left and right splits having the maximum information gain based on exchange method
        return best_partition

    def compute_information_gain(self, left_split, right_split):
        """
        Method to compute information gain using entropy for the given left and right splits
        :param left_split: list of label details in current left split
        :param right_split: list of label details in current right split
        :return:
        """
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
    def build_linear_discriminant(left_split, right_split):
        """
        Method to train and build an LDA model for the given left and right splits
        :param left_split: list of label details in the left split
        :param right_split: list of label details in the right split
        :return: trained lda_model object for the given splits
        """
        features, labels = None, None
        # Generate features and labels based on the left and right splits
        # Use 0 as the label for left split and 1 as the label for right split
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

        # Return the trained lda_model
        return LDA().fit(features, labels)

    @staticmethod
    def compute_entropy(prediction, total):
        """
        Method to compute entropy given the predicted and true label
        :param prediction: predicted label
        :param total: true label
        :return: entropy
        """
        val = float(prediction) / total
        return -1.0 * val * np.log2(val)

    @staticmethod
    def get_lda_predictions(split, lda):
        """
        Method to return predictions of a split given an lda model
        :param split: list of label details
        :param lda: trained lda_model
        :return: predicted labels
        """
        left_predictions = []
        right_predictions = []
        # For each label detail count the number of predictions of label 0 (left) and label 1 (right) made by the
        # lda model for the features in label detail
        for label_detail in split:
            left_count = 0
            right_count = 0
            rows = label_detail.sample_count
            X = label_detail.get_values()
            for i in range(rows):
                if lda.predict([X[i]]) == 0:
                    left_count += 1
                else:
                    right_count += 1

            left_predictions.append(left_count)
            right_predictions.append(right_count)

        return left_predictions, right_predictions


if __name__ == '__main__':
    # Iris Dataset
    dataset = pd.read_csv('data/iris.data')
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    model = LDTree().fit(X_train, y_train)
    print(f'Training accuracy: {accuracy_score(y_train, model.predict(X_train))}')
    print(f'Validation accuracy: {accuracy_score(y_test, model.predict(X_test))}')

    # Breast Cancer Dataset
    dataset = pd.read_csv('data/breast_cancer.data')
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    model = LDTree().fit(X_train, y_train)
    print(f'Training accuracy: {accuracy_score(y_train, model.predict(X_train))}')
    print(f'Validation accuracy: {accuracy_score(y_test, model.predict(X_test))}')

    # Ecoli Dataset
    dataset = pd.read_csv('data/ecoli.data', sep='\s+')
    X = dataset.iloc[:, 1:-1]
    y = dataset.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    model = LDTree().fit(X_train, y_train)
    print(f'Training accuracy: {accuracy_score(y_train, model.predict(X_train))}')
    print(f'Validation accuracy: {accuracy_score(y_test, model.predict(X_test))}')
