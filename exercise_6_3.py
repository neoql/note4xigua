import torch
import torch.nn
import numpy as np
import sklearn.datasets

from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn import tree


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
    clf = tree.DecisionTreeClassifier()
    y_pred = cross_val_predict(clf, X, y, cv=10)
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

    print_title('Decision tree', 'Iris')
    eval_decision_tree(iris_X, iris_y, iris.target_names)

    print_title('Decision tree', 'Wine')
    eval_decision_tree(wine_X, wine_y, wine.target_names)


if __name__ == '__main__':
    main()
