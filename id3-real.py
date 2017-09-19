import copy
import dataset
from dataset import *
from attributes import *
import sys


class DTree:
    'Represents a decision tree created with the ID3 algorithm'

    def __init__(self, classifier, training_data, attributes):
        attributes.sortValues()
        classifier.values = sorted(classifier.values)
        self.rootNode = Node(classifier, None, None, training_data, None)
        self.rootNode.attachChildren(makeTree(training_data, classifier, attributes, self.rootNode))

    def test(self, classifier, testing_data):
        num_pass = 0
        for t in testing_data:
            if self.individualTest(classifier, t):
                num_pass += 1
        return num_pass

    def individualTest(self, classifier, test):
        return helpTest(classifier, self.rootNode, test)

    def dump(self):
        output = ''
        for n in self.rootNode.children:
            output += dumpTree(n, '');
        return output


class Node:
    'Represent a part of a tree'
    def __init__(self, classifier, current_attr, current_value, current_dataset, parentNode):
        self.parentNode = parentNode
        if current_attr:
            self.attr = current_attr
        if current_value:
            self.value = current_value
        self.percent = current_dataset.getPercent(classifier)
        self.dataset = current_dataset

    def attachChildren(self, children):
        self.children = children


def makeTree(dataset, classifier, attributes, parentNode):
    if dataset.isEmpty() or len(attributes) == 0:
        result = parentValue(classifier, parentNode)
        return result

    # if there is only one classifier value in the remaining dataset,
    # return this value as the leaf
    result = dataset.checkResult(classifier)
    if result:
        return result

    # choose next attribute to split the tree
    attr = selectAttr(dataset, classifier, attributes)
    attributes.remove(attr.name)

    # make a new child tree for each of the value options for the attribute
    child_tree = []
    for v in attr.values:
        new_attributes = copy.copy(attributes)
        local_min, local_max = getRange(v)

        if (local_min or local_min == 0.0) and (local_max or local_max == 0.0):
            new_dataset = DataSet(False, False, dataset.getRealExamples(attr, local_min, local_max))
        else:
            new_dataset = DataSet(False, False, dataset.getExamples(attr, v))

        # make a new child tree
        new_child = Node(classifier, attr, v, new_dataset, parentNode)
        # recursively attach new child the tree to the current node
        new_child.attachChildren(makeTree(new_dataset, classifier, new_attributes, new_child))
        child_tree.append(new_child)
    return child_tree


# Choose the attribute that will provide the least entropy.
def selectAttr(dataset, classifier, attributes):
    if len(attributes) == 0:
        return
    if len(attributes) == 1:
        if len(attributes[0].values) == 1:
            attr, attr_entropy = getRealEntropy(dataset, classifier, attributes[0])
            return attr
        return attributes[0]
    min_entropy = sys.float_info.max
    min_entropy_attr = None

    for i in range(len(attributes)):
        if len(attributes[i].values) == 1:
            copy_attribute = Attribute(attributes[i].name, attributes[i].values)
            attr, attr_entropy = getRealEntropy(dataset, classifier, copy_attribute)
        else:
            attr, attr_entropy = attrEntropy(dataset, classifier, attributes[i])
        if attr_entropy < min_entropy:
            min_entropy = attr_entropy
            min_entropy_attr = attr

    return min_entropy_attr


# Calculate an entropy for one attribute
def attrEntropy(dataset, classifier, attr):
    attr_values = attr.values
    attr_percent = dataset.getPercent(attr)
    attr_entropy = []
    for v in attr_values:
        attr_examples = DataSet(False, False, dataset.getExamples(attr, v))
        if attr_examples.isEmpty():
            attr_entropy.append(0.0)
        else:
            attr_entropy.append(attr_examples.entropy(classifier))
    entropy = 0.0
    for i in range(len(attr_percent)):
        entropy += attr_percent[i] * attr_entropy[i]
    return attr, entropy



def getRealEntropy(dataset, classifier, attr):
    # Get a list of possible values of the attribute
    values = []
    for d in dataset:
        values.append(d.get_value(attr))
    value_list = sorted(list(set(values)))

    local_min, local_max = getRange(attr.values[0])
    if not local_min:
        local_min = -(sys.float_info.max-1)

    if not local_max:
        local_max = sys.float_info.max

    # Find the cutoff that maximize the entropy
    min_entropy = sys.float_info.max
    min_cutoff = -1
    for i in range(len(value_list)-1):
        cutoff = float(value_list[i])
        local_min, local_max = getRange(attr.values[0])

        sub_dataset = []
        sub_dataset.append(DataSet(False, False, dataset.getRealExamples(attr, local_min, cutoff)))
        sub_dataset.append(DataSet(False, False, dataset.getRealExamples(attr, cutoff, local_max)))

        entropy = 0.0
        for d in sub_dataset:
            entropy += (float(len(d))/len(dataset)) * d.entropy(classifier)

        if entropy < min_entropy:
            min_entropy = entropy
            min_cutoff = cutoff

    # Find the smalles value that's greater than the cutoff value
    next_cutoff = 0
    for i in range(len(value_list)):
        if float(value_list[i]) == float(min_cutoff):
            if i == len(value_list) - 1:
                next_cutoff = value_list[i]
            else:
                next_cutoff = value_list[i+1]
            break

    # Calculate the middle point
    new_cutoff = (float(min_cutoff) + float(next_cutoff)) / 2

    # Provide 'categorical' values for the attribute
    new_value = []
    new_value.append(str(local_min) + '..' + str(new_cutoff))
    new_value.append(str(new_cutoff) + '..' + str(local_max))
    attr.values = new_value
    return attr, min_entropy


# Get the float values from a string range
def getRange(str_range):
    value_range = str_range.split('..')
    if len(value_range) != 2:
        return None, None
    local_min = float(value_range[0])
    local_max = float(value_range[1])
    return local_min, local_max


# Recursively looking for a leaf value from the parent.
def parentValue(classifier, parentNode):
    # Return the alphabetically smallest value if it's at the root
    if not parentNode:
        return classifier.values[0]

    class_percent = parentNode.percent
    class_values = classifier.values

    max_percent = max(class_percent)

    max_value = None
    for i in range(len(class_percent)):
        if class_percent[i] == max_percent:
            if max_value == None:
                max_value = class_values[i]
            else:
                # Recursively looking at the parent when no value is more frequent than the others
                return parentValue(classifier, parentNode.parentNode)
    return max_value


def dumpTree(node, space):
    if isinstance(node.children, str):
        output = space + node.attr.name + ':' + node.value + ', ' + str(len(node.dataset)) + '\n'
        output += space + ' <' + node.children + '>\n'
        return output
    else:
        output = space + node.attr.name + ':' + node.value + ', ' + str(len(node.dataset)) + '\n'
        newSpace = space + ' ';
        for n in node.children:
            output += dumpTree(n, newSpace)
        return output


def helpTest(classifier, node, test):
    if isinstance(node.children, str):
        answer = test.get_value(classifier)
        return answer == node.children

    next_nodes = node.children
    next_attr = node.children[0].attr
    test_value = test.get_value(next_attr)
    for child in next_nodes:
        if matchValue(child.value, test_value):
            return helpTest(classifier, child, test)
    return False


# Check if value_2 meets the requirement of value_1
def matchValue(value_1, value_2):
    local_min, local_max = getRange(value_1)
    if (local_min or local_min == 0.0) and (local_max or local_max == 0.0):
        value = float(value_2)
        if value > local_min and value <= local_max:
            return True
        return False
    elif value_1 == value_2:
        return True
    return False






