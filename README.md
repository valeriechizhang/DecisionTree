# DecisionTree
Demonstrate a decision tree that uses the ID3 algorithm.

### Tool: Python 2.7.5

### Description:
The program accepts a set of data in .csv, and from that data produce a decision tree. It will also allow testing the accuracy of the decision tree by running test cases
against the tree and comparing the decision tree’s performance against the known classifications.


id3.py: an decision tree program that only considers integer values

id3-real.py: an decision tree program that considers real number values

main.py: Provides a command-line interface to the decision tree. It takes positional parameters for the decision tree
algorithm module name, and for the name of the classification attribute. Invoke with --help to see complete list of options.


### How to run:
Run individual tests:

./main.py id3 [classification attributes] --attributes [attributes txt file] --train [train csv file] --test [test csv file]

Run all tests:

./run_tests.sh tests OR ./run_tests.sh id3-real-tests


### Reference:

Santa Clara University COEN266 Artificial Intelligence
