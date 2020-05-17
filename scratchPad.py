#!/usr/bin/env python
# coding: utf-8

# In[152]:


import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class Utils:
    def load_dataset(self, path, cols):
        df = pd.read_csv(path, names=cols)
        return df


def euclidean_distance(X1, X2):
    X1, X2 = np.array(X1), np.array(X2)

    return np.sqrt(np.sum(np.square(X1 - X2)))

def TreeNode():

    def __init__(self):
        self.val = None
        self.left = None
        self.right = None
        
    def __init__(self, lda, left, right):
        self.val = lda
        self.left = left
        self.right = right
        
    def __init__(self, label):
        self.val = label
        self.left = None
        self.right = None
        
    def is_leaf(self):
        return self.left is None and self.right is None


class LabelDetails:
    def __init__(self, X, y, label):
        self.X = X
        self.y = y
        self.label = label
        self.compute_means()
        self.sample_count = X.shape[0]
        
    def compute_means(self):
        self.mean = self.X.describe().loc['mean'].values.reshape(-1,1)
        
    def get_values(self):
        return self.X.values
    
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, LabelDetails):
            return self.label == other.label and (self.mean == other.mean).all()
        return False
    
    def __ne__(self, other):
        ret = not self.__eq__(other)
        return ret
        
    def __str__(self):
        str_ = "Class details: "
        str_ += " - means:"+str(self.mean) + "\n"
        str_ += " - class_name:"+self.label + "\n"
        str_ += " - num_instances:"+str(self.samples)
        return str_
    

class Model:
    def __init__(self):
        # Initialize class parameters
        self.root = TreeNode()
        self.distance_fn = Utils().euclidean_distance
    
    def fit(self, X, y):
        # Function to train a decision tree for the given training data X
        label_details = self.create_labels(X, y)
        self.root = self.build_tree(label_details)
        print(str(label_details))
        
    def predict(self, test_sample):
        # Function to classify test data
        pass
    
    def build_linear_discriminant(self, left_split, right_split):
        features = None

        for label_detail in left_split:
            if features is None:
                features = label_detail.get_values()
                labels = np.zeros(features.shape[0]) # 0.0.0.0
            else:
                features = np.append(features, label_detail.get_values())
                labels = np.append(labels, np.zeros(features.shape[0]))

        for label_detail in right_split:
            if features is None:
                features = label_detail.get_values()
                labels = np.ones(features.shape[0]) # 1.1.1.1.
            else:
                features = np.append(features, label_detail.get_values())
                labels = np.append(labels, np.ones(features.shape[0]))

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

    
    def heuristic_split(self, label_details):
        maximum_distance = float('-inf')
        splits = {}
        for i in range(len(label_details)):
            for j in range(i, len(label_details)):
                distance = euclidean_distance(label_details[i].mean, label_details[j].mean)
                if distance > maximum_distance:
                    maximum_distance = distance
                    splits = {'left': [label_details[i]], 'right': [label_details[j]]}
        maximum_distance_splits = splits['left'] + splits['right']
        for label_detail in label_details:
            if label_detail not in maximum_distance_splits:
                left_distance = euclidean_distance(label_detail.mean, splits['left'][0].mean)
                right_distance = euclidean_distance(label_detail.mean, splits['right'][0].mean)
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
            
            # equals()
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

        # TODO: remove this
        assert(len(best_partition[0]) + len(best_partition[1]) == len(left_split) + len(right_split))
        return best_partition 
        
    
    def compute_information_gain(self, left_split, right_split):
        if len(left_split) == 0 or len(right_split) == 0:
            return float('-inf')
        
        lda_model = self.build_linear_discriminant(left_split, right_split)
        
        total_left_samples = sum([ld.sample_count for ld in left_split])
        total_right_samples = sum([rd.sample_count for rd in right_split])
        total = total_left_samples + total_right_samples
#         print(str(total) + " " + str(total_right_samples) + " " + str(total_left_samples))

        e0 = 0.0
        information_gain = 0.0
        for detail in left_split + right_split:
            e0 = e0 + self.compute_entropy(detail.sample_count, total)
        
        left_predictions = []
        right_predictions = []
        
        for split in [left_split, right_split]:
            l, r = self.get_lda_predictions(split, lda)
            left_predictions = left_predictions + l
            right_predictions = right_predictions + r
            
        
        information_gain = e0
        for predictions, total_count in [(left_predictions, total_left_samples), (right_predictions, total_right_samples)]:
            val = 0.0
            for prediction in predictions:
                if prediction != 0.0:
                    temp = float(prediction)/total_count
                    val = val + temp * np.log2(temp)
            information_gain += val * total_count/total
        
        return information_gain
    
        
    def compute_entropy(self, prediction, total):
        val = float(prediction)/total
        return -1.0 * (val) * np.log2(val)
            
        
    def get_lda_predictions(split, lda):
        left_predictions = []
        right_predictions = []
        for label_detail in split:
            left_count = 0
            right_count = 0
            rows = label_detail.X.shape[0]
            X = label_detail.X.values
            for i in range(rows):
                if lda.predict([ X[i] ]) == 0:
                    left_count += 1
                else:
                    right_count += 1
            
            left_predictions.append(left_count)
            right_predictions.append(right_count)
            
        return left_predictions, right_predictions
        
    
    def create_labels(self, X, y):
        unique_classes = set(y.unique())
        label_details = []
        for class_type in unique_classes:
            yi = y[y==class_type]
            indexes = yi.index
            xi = X.iloc[indexes]
            label_detail = LabelDetails(xi, yi, class_type)
            
            label_details.append(label_detail)
        
        return label_details


class Driver:
    def main(self):
        pass






model = Model()
ret = model.fit(X, y)
model.p