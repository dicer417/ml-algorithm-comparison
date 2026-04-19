import numpy as np
import math
import random

import pandas as pd


class Node:
    """
    A class to represent a node in a neural network
    """
    def __init__(self, node_type, num_weights, node_num=None, parent_nodes=None, output_class=None):
        """
        Initializes a Node object

        :param node_type: (String) Either 'hidden' or 'output'
        :param num_weights: (Int) Number of edges entering the Node, i.e. the number of parents
        :param node_num: (Int) Unique Node identifier number
        :param parent_nodes: (Node) Vector of Nodes representing the Node's parents in the neural network
        :param output_class: (String) For classification, the class value to which the output node pertains
        """
        self.nodeNum = node_num             # Unique identifier for each node
        self.type = node_type               # Hidden or output
        self.lastOutputValue = None         # Save the last value output from the node
        self.lastInputValues = None         # Save the last vector of inputs provided to the node

        # List of weights for incoming inputs; randomize to small values initially
        self.weights = [random.uniform(-0.01, 0.01) for _ in range(num_weights)]

        # Track the parents and children of each node
        self.parentNodes = parent_nodes
        self.childNodes = []

        # For classification, the class to which the output node corresponds
        self.outputClass = output_class

    def return_node(self, level=None, convert_weights_to_strings=False):
        """
        Transforms the information in a Node into a readable DataFrame

        :param level: (Int) The layer level of the Node in the neural network
        :param convert_weights_to_strings: (Boolean) Whether to convert the numeric vector of weights to strings
        :return: (DataFrame) The Node's information in DataFrame form
        """
        # Extract the parent nodes
        parent_node_nums = []
        if self.parentNodes is not None:
            for parentNode in self.parentNodes:
                parent_node_nums.append(parentNode.nodeNum)

        # Extract the child nodes
        child_node_nums = []
        if self.childNodes is not None:
            for childNode in self.childNodes:
                child_node_nums.append(childNode.nodeNum)

        # Convert the weight vector to strings
        if convert_weights_to_strings:
            weights = [str(x) for x in self.weights]
        else:
            weights = self.weights

        # Create the DataFrame
        node_dt = pd.DataFrame({
            'nodeNum': [self.nodeNum],
            'type': [self.type],
            'lastOutputValue': [self.lastOutputValue],
            'lastInputsValues': [self.lastInputValues],
            'weights': [weights],
            'parentNodes': [parent_node_nums],
            'childNodes': [child_node_nums],
            'outputClass': [self.outputClass],
            'level': level
        })

        return node_dt

    def calc_weighted_sum_of_inputs(self, inputs):
        """
        Multiply a vector of inputs with the Node's input weights

        :param inputs: (numeric) Vector of inputs into the Node
        :return: (numeric) The weighted sum of the inputs
        """
        # Add a term for the bias
        input_vec = [1] + inputs

        # Return the weighted sum
        return np.matmul(input_vec, self.weights)

    def calc_sigmoid(self, inputs):
        """
        Calculates the sigmoid function for a vector of inputs

        :param inputs: (numeric) A vector of inputs
        :return: (numeric) The sigmoid value of the weighted sum of the inputs
        """
        # Calculate the weighted sum of the inputs
        x_val = self.calc_weighted_sum_of_inputs(inputs=inputs)

        # Handling extreme cases for overflow
        if math.isinf(x_val):
            z_val = 0
        else:
            try:
                # Sigmoid function
                z_val = 1 / (1 + math.exp(-1 * x_val))
            except OverflowError:
                print(x_val)
                z_val = 1

        return z_val

    def update_weights(self, node_error, learning_rate):
        """
        Implements backpropagation. Takes the error attributable to the Node and recursively
        updates the weights of the Node's parents before updating its own

        :param node_error: (numeric) The error attributable to the node. For an output node, it is the full error.
        For a hidden node, it is the error of the previous node multiplied by the weight that node attributes to it.
        :param learning_rate: (numeric) The learning rate to apply
        :return: None
        """

        # First, update the weights of the parent's nodes
        weight_ind = 1
        if self.parentNodes is not None:
            for parentNode in self.parentNodes:
                parent_node_error = node_error * self.weights[weight_ind]

                parentNode.update_weights(node_error=parent_node_error, learning_rate=learning_rate)

                weight_ind = weight_ind + 1

        # Print for the video
        print("Updating weights on node #" + str(self.nodeNum) + "--------")
        print("Old weights")
        print(self.weights)

        # If the node is an output node, update by dw = -learning_rate * (r - y) * z
        if self.type == 'output':
            print("Output node's error: " + str(node_error))
            weight_changes = [learning_rate * node_error * lastInputValue for lastInputValue in self.lastInputValues]
        else:
            print("Node's error from this parent: " + str(node_error))
            weight_changes = [learning_rate * node_error * self.lastOutputValue * (1 - self.lastOutputValue) *
                              lastInputValue for lastInputValue in self.lastInputValues]


        # Apply the weights changes to the weights
        new_weights = []
        for ind in list(range(len(self.weights))):
            new_weights.append(self.weights[ind] + weight_changes[ind])
        self.weights = new_weights

        print("New weights")
        print(self.weights)


if __name__ == '__main__':
    print(np.matmul([1, 2], [1, 2]))


