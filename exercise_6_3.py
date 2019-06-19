import torch
import torch.nn
import numpy as np
import sklearn.datasets

from math import log2

from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report


class BPNN(object):
    def __init__(self, input, hidden, output):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input, hidden),
            torch.nn.Linear(hidden, output),
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

    def fit(self, X, y):
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, X):
        with torch.no_grad():
            y_pred = self.model(X)
        return torch.argmax(y_pred, dim=1)


class DecisionTree(object):
    def __init__(self, algorithm='C4.5'):
        self.algorithm = algorithm
        self.root = None

    def predict(self, x):
        y = np.zeros([x.shape[0]])

        for i, xx in enumerate(x):
            node = self.root

            while not node.is_leaf:
                node = node.left if xx[node.attr] < node.val else node.right

            y[i] = node.label

        return y

    def fit(self, x, y):
        self.root = self.gen_node(x, y)

    def gen_node(self, x, y):
        y_val_count = self.value_count(y)

        node_label = max(y_val_count, key=y_val_count.get)

        node = Node(node_label)
        if len(y_val_count) == 1:
            return node

        attr, val = self.select_attr(x, y)
        if attr is None:
            return node

        attr_val = x[:, attr]
        left = attr_val < val
        right = attr_val >= val

        node.attr = attr
        node.val = val
        node.left = self.gen_node(x[left], y[left])
        node.right = self.gen_node(x[right], y[right])

        return node

    def select_attr(self, x, y, y_val_count=None):
        n = len(y)

        if y_val_count is None:
            y_val_count = self.value_count(y)

        ent = self.entropy(y, y_val_count)

        attr_count = x.shape[1]
        radios = np.zeros((attr_count,))
        select_vals = np.zeros((attr_count,))
        for i in range(attr_count):
            x_attr = x[:, i]
            vals = np.sort(x_attr)
            if vals[0] == vals[-1]:
                return None, None

            dis = {vals[0]}
            max_radio = 0
            select_val = -1
            for val in vals:
                if val in dis:
                    continue
                dis.add(val)

                left = y[x_attr < val]
                right = y[x_attr >= val]

                gain = ent
                gain -= (len(left) / n) * self.entropy(left)
                gain -= (len(right) / n) * self.entropy(right)

                iv = -((len(left)/n)*log2(len(left)/n) + (len(right)/n)*log2(len(right)/n))
                radio = gain / iv
                if radio > max_radio:
                    max_radio = radio
                    select_val = val
            radios[i] = max_radio
            select_vals[i] = select_val
        attr = np.argmax(radios)
        return attr, select_vals[attr]

    @staticmethod
    def value_count(values):
        value_cont = {}

        for val in values:
            if val in value_cont:
                value_cont[val] += 1
            else:
                value_cont[val] = 1

        return value_cont

    @staticmethod
    def entropy(values, value_count=None):
        ent = 0

        total = len(values)
        if value_count is None:
            value_count = DecisionTree.value_count(values)

        for _, count in value_count.items():
            p = count / total
            ent -= (p * log2(p))

        return ent


class Node(object):
    def __init__(self, label):
        self.label = label
        self.left = None
        self.right = None
        self.attr = -1
        self.val = -1

    @property
    def is_leaf(self):
        return self.left is None


def eval_rbf_svc(X, y, names):
    svc = svm.SVC(gamma='scale')

    y_pred = cross_val_predict(svc, X, y, cv=10)
    print(classification_report(y, y_pred, target_names=names))


def eval_linear_svc(X, y, names):
    svc = svm.SVC(kernel='linear')

    y_pred = cross_val_predict(svc, X, y, cv=10)
    print(classification_report(y, y_pred, target_names=names))


def eval_bp_nn(X, y, names):
    epoch = 200

    input = X.shape[1]
    output = 3
    hidden = input * 2 - 1

    kf = KFold(n_splits=10)

    y_pred = np.array([])

    for train, test in kf.split(X, y):
        clf = BPNN(input, hidden, output)

        clf.model.train()
        for _ in range(epoch):
            clf.fit(torch.from_numpy(X[train]).float(), torch.from_numpy(y[train]))

        clf.model.eval()
        y_pred = np.append(y_pred, clf.predict(torch.from_numpy(X[test]).float()).numpy())

    print(classification_report(y, y_pred, target_names=names))


def eval_decision_tree(X, y, names):
    kf = KFold(n_splits=10)
    y_pred = np.array([])

    for train, test in kf.split(X, y):
        clf = DecisionTree()
        clf.fit(X[train], y[train])

        y_pred = np.append(y_pred, clf.predict(X[test]))

    print(classification_report(y, y_pred, target_names=names))


def print_title(model_name, dataset_name):
    print('---------------------------------------------------------')
    print('{}-{}'.format(model_name, dataset_name))
    print('---------------------------------------------------------')


def main():
    iris = sklearn.datasets.load_iris()
    wine = sklearn.datasets.load_wine()

    iris_X, iris_y = shuffle(iris.data, iris.target)
    wine_X, wine_y = shuffle(wine.data, wine.target)

    print_title('SVM(RBF kernel)', 'Iris')
    eval_rbf_svc(iris_X, iris_y, iris.target_names)

    print_title('SVM(RBF kernel)', 'Wine')
    eval_rbf_svc(wine_X, wine_y, wine.target_names)

    print_title('SVM(Linear kernel)', 'Iris')
    eval_linear_svc(iris_X, iris_y, iris.target_names)

    print_title('SVM(Linear kernel)', 'Wine')
    eval_linear_svc(wine_X, wine_y, wine.target_names)

    print_title('BP Neural Network', 'Iris')
    eval_bp_nn(iris_X, iris_y, iris.target_names)

    print_title('BP Neural Network', 'Wine')
    eval_bp_nn(wine_X, wine_y, wine.target_names)

    print_title('Decision tree(C4.5)', 'Iris')
    eval_decision_tree(iris_X, iris_y, iris.target_names)

    print_title('Decision tree(C4.5)', 'Wine')
    eval_decision_tree(wine_X, wine_y, wine.target_names)


if __name__ == '__main__':
    main()
