import math
import re
import sys


class Example:
  'An individual example with values for each attribute'

  def __init__(self, values, attributes, filename, line_num):
    if len(values) != len(attributes):
      sys.stderr.write(
        "%s: %d: Incorrect number of attributes (saw %d, expected %d)\n" %
        (filename, line_num, len(values), len(attributes)))
      sys.exit(1)
    # Add values, Verifying that they are in the known domains for each
    # attribute
    self.values = {}
    for ndx in range(len(attributes)):
      value = values[ndx]
      attr = attributes.attributes[ndx]
      if value not in attr.values and not matchValues(attr.values, value):
        sys.stderr.write(
          "%s: %d: Value %s not in known values %s for attribute %s\n" %
          (filename, line_num, value, attr.values, attr.name))
        sys.exit(1)
      self.values[attr.name] = value

  # Find a value for the specified attribute, which may be specified as
  # an Attribute instance, or an attribute name.
  def get_value(self, attr):
    if isinstance(attr, str):
      return self.values[attr]
    else:
      return self.values[attr.name]

def matchValues(values, value):
  if len(values) == 1:
    return True
  for v in values:
    if matchValue(v, value):
      return True
  return False

def matchValue(value_1, value_2):
  local_min, local_max = getRange(value_1)
  if (local_min or local_min == 0.0) and (local_max or local_max == 0.0):
    value = float(value_2)
    if value >= local_min and value <= local_max:
      return True
    return False
  elif value_1 == value_2:
    return True
  return False

def getRange(str_range):
  value_range = str_range.split('..')
  if len(value_range) != 2:
    return None, None
  local_min = float(value_range[0])
  local_max = float(value_range[1])
  return local_min, local_max


class DataSet:
  'A collection of instances, each representing data and values'

  def __init__(self, data_file=False, attributes=False, examples=False):
    self.all_examples = []
    self.attributes = attributes
    if data_file:
      line_num = 1
      num_attrs = len(attributes)
      for next_line in data_file:
        next_line = next_line.rstrip()
        next_line = re.sub(".*:(.*)$", "\\1", next_line)
        attr_values = next_line.split(',')
        new_example = Example(attr_values, attributes, data_file.name, line_num)
        self.all_examples.append(new_example)
        line_num += 1
    elif examples:
      self.all_examples = examples[:]


  def __len__(self):
    return len(self.all_examples)

  def __getitem__(self, key):
    return self.all_examples[key]

  def append(self, example):
    self.all_examples.append(example)

  # Get a set of examples with a certain value for an attribute
  def getExamples(self, attr, value):
    examples = []
    for e in self.all_examples:
      if e.get_value(attr) == value:
        examples.append(e)
    return examples

  # Get a set of examples within a certain value range for an attribute
  def getRealExamples(self, attr, min_value, max_value):
    examples = []
    for e in self.all_examples:
      val = float(e.get_value(attr))
      if val > min_value and val <= max_value:
        examples.append(e)
    return examples

  def isEmpty(self):
    if len(self.all_examples) == 0:
      return True
    return False

  # Check if all the examples in the dataset have the same value for the classifier
  def checkResult(self, classifier):
    if len(self.all_examples) == 0:
      return False
    value = self.all_examples[0].get_value(classifier)
    for e in self.all_examples[1:]:
      if e.get_value(classifier) != value:
        return False
    return value

  # Calculate the percentage of examples that have each of the classifier values
  def getPercent(self, classifier):
    class_values = classifier.values
    class_percents = []
    total = float(len(self.all_examples))
    for v in class_values:
      count = 0
      for e in self.all_examples:
        if e.get_value(classifier) == v:
          count += 1
      if count > 0 and total > 0:
        p = float(count) / total
        class_percents.append(p)
      else:
        class_percents.append(0.0)
    return class_percents


  # Determine the entropy of a collection with respect to a classifier.
  # An entropy of zero indicates the collection is completely sorted.
  # An entropy of one indicates the collection is evenly distributed with
  # respect to the classifier.
  def entropy(self, classifier):
    entropy = 0.0
    class_percent = self.getPercent(classifier)
    for p in class_percent:
      if p == 0.0:
        continue
      entropy -= p * math.log(p, 2.0)
    return entropy


