from math import log2

import pandas as pd
from pydotplus import graphviz


class DecisionTreeClassifier(object):
    def __init__(self, criterion="ID3"):
        self.__root = None
        self.__criterion = criterion

    def fit(self, features, labels):
        self.__root = self.__tree_generate(features, labels)

    def __tree_generate(self, features, labels):
        node = Node()

        label_count = self.__value_count(labels)

        node.label = max(label_count, key=label_count.get)
        if len(label_count) == 1 or len(features.columns) == 0:
            return node

        opt_attr, div_value = self.__optimal_attr(features, labels)

        if opt_attr is None:
            return node

        node.attr = opt_attr
        feature = features[opt_attr]
        feature_option = set(feature)
        if div_value is None:
            features = features.drop(opt_attr, axis=1)
            for val in feature_option:
                sub_features = features[feature == val]
                sub_labels = labels[feature == val]
                node.branches[val] = self.__tree_generate(sub_features, sub_labels)
        else:
            left_features = features[feature < div_value]
            left_labels = labels[feature < div_value]
            node.branches["<{:.3f}".format(div_value)] = self.__tree_generate(left_features, left_labels)

            right_features = features[feature > div_value]
            right_labels = labels[feature > div_value]
            node.branches[">{:.3f}".format(div_value)] = self.__tree_generate(right_features, right_labels)

        return node

    def predict(self, feature):
        pass

    @staticmethod
    def __value_count(values):
        label_count = {}

        for label in values:
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1

        return label_count

    def __optimal_attr(self, features, labels):
        return self.__optimal_attr_by_information_gain(features, labels)

    @staticmethod
    def __optimal_attr_by_information_gain(features, labels):
        ent = DecisionTreeClassifier.__information_entropy(labels)

        attr_gain = {}
        attr_div = {}

        for attr in features.columns:
            feature = features[attr]
            feature_count = DecisionTreeClassifier.__value_count(feature)
            n = len(feature)

            if len(feature_count) == 1:
                continue

            if feature.dtype == float or feature.dtype == int:
                vals = list(feature_count)
                vals.sort()

                div_gain = {}

                for i in range(len(vals) - 1):
                    div = (vals[i] + vals[i + 1]) / 2
                    left = labels[feature < div]
                    right = labels[feature > div]

                    gain = ent

                    gain -= (len(left) / n) * DecisionTreeClassifier.__information_entropy(left)
                    gain -= (len(right) / n) * DecisionTreeClassifier.__information_entropy(right)

                    div_gain[div] = gain

                div = max(div_gain, key=div_gain.get)
                attr_gain[attr] = div_gain[div]
                attr_div[attr] = div
            else:
                gain = ent

                for val, count in feature_count.items():
                    gain -= ((count / n) * DecisionTreeClassifier.__information_entropy(labels[feature == val]))
                attr_gain[attr] = gain
                attr_div[attr] = None

        try:
            attr = max(attr_gain, key=attr_gain.get)
        except ValueError:
            return None, None

        return attr, attr_div[attr]

    @staticmethod
    def __information_entropy(labels):
        ent = 0

        total = len(labels)
        label_count = DecisionTreeClassifier.__value_count(labels)

        for _, count in label_count.items():
            p = count / total
            ent -= (p * log2(p))

        return ent

    def graph(self):
        g = graphviz.Dot()
        self.__seq = 0
        self.__fill_graph(self.__root, None, "", g)
        return g.to_string()

    def __fill_graph(self, node, father, branch, g):
        if node.attr is None:
            title = "好瓜：{}".format(node.label)
        else:
            title = "属性：{}".format(node.attr)

        g_node = graphviz.Node(self.__seq, label=title)
        g.add_node(g_node)
        if father is not None:
            g.add_edge(graphviz.Edge(father, g_node, label=branch))

        for val, child in node.branches.items():
            self.__seq += 1
            self.__fill_graph(child, g_node, val, g)


class Node(object):
    def __init__(self, attr=None, label=None, branches=None):
        if branches is None:
            branches = {}
        self.attr = attr
        self.label = label
        self.branches = branches


def main():
    with open('dataset/xigua_dataset_3_0.csv') as fd:
        df = pd.read_csv(fd)

    features = df[df.columns[1:-1]]
    labels = df[df.columns[-1]]

    clf = DecisionTreeClassifier()
    clf.fit(features, labels)

    g = graphviz.graph_from_dot_data(clf.graph())
    g.write_png('tree.png')


if __name__ == '__main__':
    main()
