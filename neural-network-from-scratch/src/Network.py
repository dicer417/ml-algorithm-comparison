import pandas as pd
import os
import math
import numpy as np
import ast  # This is necessary for checkpointing and does not aid in the development of the models

from src.Node import Node
from src.DataLoader import DataLoader


class Network:
    """
    A class to represent a feedforward linear Neural Network with 0, 1, or 2 hidden layers
    """
    def __init__(self, data_loader, is_autoencoder=False, size_hidden_layer_1=None, size_hidden_layer_2=None):
        """
        Initializes a new neural Network object

        :param data_loader: (DataLoader) Object contains the data with which to train the neural network
        :param is_autoencoder: (Boolean) Whether the Network is an autoencoder
        :param size_hidden_layer_1: (int) Number of nodes in the first hidden layer. None if no hidden layers
        :param size_hidden_layer_2: (int) Number of nodes in the second hidden layer. None if no second hidden layer
        """
        # Inherit attributes from the DataLoader
        self.dataLoader = data_loader
        self.training_data = data_loader.trainingData
        self.predictor = data_loader.predictor

        self.is_autoencoder = is_autoencoder                # Whether the network is an autoencoder

        # Save a vector of the input names --> necessary because of one-hot coding
        non_feature_cols = [self.predictor, 'sampleNum', 'set']
        self.inputNames = [col for col in self.training_data.columns if col not in non_feature_cols]

        # Save a depth of tree value
        self.depth = 1

        # Initialize information about the output nodes. Update if hidden layers are present
        output_nodes_parents = None
        output_nodes_num_parents = len(self.inputNames)

        # Track the number of nodes --> so we can assign a unique number to each node
        current_node_num = 1

        # Create hidden layer 1
        if size_hidden_layer_1 is not None:
            self.hidden_layer_1_nodes = []
            for node_num in list(range(size_hidden_layer_1)):
                self.hidden_layer_1_nodes.append(Node(node_type='hidden', num_weights=len(self.inputNames) + 1,
                                                      node_num=current_node_num, parent_nodes=None,
                                                      output_class=None))
                current_node_num = current_node_num + 1
            self.depth = self.depth + 1
            output_nodes_parents = self.hidden_layer_1_nodes
            output_nodes_num_parents = size_hidden_layer_1
        else:
            self.hidden_layer_1_nodes = None

        # Create hidden layer 2
        if size_hidden_layer_2 is not None:
            self.hidden_layer_2_nodes = []
            for node_num in list(range(size_hidden_layer_2)):
                self.hidden_layer_2_nodes.append(Node(node_type='hidden', num_weights=size_hidden_layer_1 + 1,
                                                      node_num=current_node_num, parent_nodes=self.hidden_layer_1_nodes,
                                                      output_class=None))
                current_node_num = current_node_num + 1
            self.depth = self.depth + 1
            output_nodes_parents = self.hidden_layer_2_nodes
            output_nodes_num_parents = size_hidden_layer_2

            # Update childNodes for hiddenLayer1
            for uniqueNode in self.hidden_layer_1_nodes:
                uniqueNode.childNodes = self.hidden_layer_2_nodes
        else:
            self.hidden_layer_2_nodes = None

        # Set output nodes
        if is_autoencoder:
            output_nodes = []
            for uniqueFeature in self.inputNames:
                output_nodes.append(Node(node_type='output', num_weights=output_nodes_num_parents + 1,
                                         node_num=current_node_num,
                                         parent_nodes=output_nodes_parents, output_class=uniqueFeature))
                current_node_num = current_node_num + 1
        else:
            if data_loader.dataset in data_loader.categorizationSets:
                output_nodes = []
                for uniquePredictorValue in self.training_data[self.predictor].unique():
                    output_nodes.append(Node(node_type='output', num_weights=output_nodes_num_parents + 1,
                                             node_num=current_node_num, parent_nodes=output_nodes_parents,
                                             output_class=uniquePredictorValue))
                    current_node_num = current_node_num + 1
            else:
                output_nodes = [Node(node_type='output', num_weights=output_nodes_num_parents + 1,
                                     node_num=current_node_num, parent_nodes=output_nodes_parents, output_class=None)]
        self.output_nodes = output_nodes

        # Update childNodes of previous layer, if there is one
        if self.hidden_layer_2_nodes is not None:
            for uniqueNode in self.hidden_layer_2_nodes:
                uniqueNode.childNodes = self.output_nodes
        elif self.hidden_layer_1_nodes is not None:
            for uniqueNode in self.hidden_layer_1_nodes:
                uniqueNode.childNodes = self.output_nodes

    def return_nodes(self, convert_weights_to_strings=False):
        """
        Converts the Nodes of the Network into a DataFrame

        :param convert_weights_to_strings: (Boolean) Whether to convert the numeric weights of the Nodes into Strings
        for easier saving
        :return: (DataFrame) A table of the Network's Nodes
        """
        # Initialize the table to hold the Nodes
        all_nodes_dt = pd.DataFrame()

        # Track the level of the hidden layers
        curr_level = 1

        # Extract the first layer of hidden nodes
        if self.hidden_layer_1_nodes is not None:
            for hiddenNode1 in self.hidden_layer_1_nodes:
                node_dt = hiddenNode1.return_node(level=curr_level,
                                                  convert_weights_to_strings=convert_weights_to_strings)
                all_nodes_dt = pd.concat(objs=[all_nodes_dt, node_dt], axis=0, ignore_index=True)
            curr_level = curr_level + 1

        # Extract the second layer of hidden nodes
        if self.hidden_layer_2_nodes is not None:
            for hiddenNode2 in self.hidden_layer_2_nodes:
                node_dt = hiddenNode2.return_node(level=curr_level,
                                                  convert_weights_to_strings=convert_weights_to_strings)
                all_nodes_dt = pd.concat(objs=[all_nodes_dt, node_dt], axis=0, ignore_index=True)
            curr_level = curr_level + 1

        # Extract the output nodes
        for outputNode in self.output_nodes:
            node_dt = outputNode.return_node(level=curr_level, convert_weights_to_strings=convert_weights_to_strings)
            all_nodes_dt = pd.concat(objs=[all_nodes_dt, node_dt], axis=0, ignore_index=True)

        return all_nodes_dt

    def save_nodes(self, file_location):
        """
        Save a table of the Network's nodes to a specific location

        :param file_location: (String) Location to save the table
        :return: None
        """
        node_dt = self.return_nodes(convert_weights_to_strings=False)
        node_dt.to_csv(file_location)

    def overwrite_nodes_from_dt(self, node_table=None, file_location=None):
        """
        Overwrites the Network's nodes with information from a table

        :param node_table: (DataFrame) A table of Node information
        :param file_location: (String) Location from which to read a table of Node information if none is provided
        :return: None
        """
        # Read in nodes if desired
        if file_location is not None:
            node_table = pd.read_csv(file_location)
            node_table['weights'] = node_table['weights'].map(ast.literal_eval)

        # Initialize vectors to hold the nodes
        hidden_nodes_1 = []
        hidden_nodes_2 = []
        output_nodes = []

        for row_ind in list(range(len(node_table))):
            # Extract values for the node
            node_row = node_table.iloc[[row_ind]]
            node_num = node_row['nodeNum'].iloc[0]
            node_level = node_row['level'].iloc[0]
            node_type = node_row['type'].iloc[0]
            node_weights = node_row['weights'].iloc[0]
            node_output_class = node_row['outputClass'].iloc[0]

            # Create new node
            new_node = Node(node_type=node_type, num_weights=len(node_weights) + 1, node_num=node_num,
                            output_class=node_output_class)
            new_node.weights = np.array(node_weights, dtype=np.float64)

            # Append the node to the correct level
            if node_type == 'hidden' and node_level == 1:
                hidden_nodes_1.append(new_node)
            elif node_type == 'hidden' and node_level == 2:
                hidden_nodes_2.append(new_node)
            else:
                output_nodes.append(new_node)

            # Update parent and child nodes and assign to network attributes
            if len(hidden_nodes_1) > 0:
                if len(hidden_nodes_2) > 0:
                    for hiddenNode1 in hidden_nodes_1:
                        hiddenNode1.childNodes = hidden_nodes_2
                    for outputNode in output_nodes:
                        outputNode.parentNodes = hidden_nodes_2
                    for hiddenNode2 in hidden_nodes_2:
                        hiddenNode2.childNodes = output_nodes
                        hiddenNode2.parentNodes = hidden_nodes_1
                else:
                    for hiddenNode1 in hidden_nodes_1:
                        hiddenNode1.childNodes = output_nodes
                    for outputNode in output_nodes:
                        outputNode.parentNodes = hidden_nodes_1

            # Set as attributes
            depth = 1
            if len(hidden_nodes_1) > 0:
                self.hidden_layer_1_nodes = hidden_nodes_1
                depth = depth + 1
            else:
                self.hidden_layer_1_nodes = None

            if len(hidden_nodes_2) > 0:
                self.hidden_layer_2_nodes = hidden_nodes_2
                depth = depth + 1
            else:
                self.hidden_layer_2_nodes = None
            self.output_nodes = output_nodes
            self.depth = depth

    def append_autoencoder_to_network(self, size_hidden_layer_2):
        """
        Takes an autoencoder Network, removes its output layer, and attaches it to a new hidden layer and output layer

        :param size_hidden_layer_2: (int) Number of nodes to put in the new hidden layer
        :return: None
        """
        # Bail if this isn't an autoencoder
        if not self.is_autoencoder:
            return

        self.is_autoencoder = False

        current_node_num = len(self.hidden_layer_1_nodes) + 1
        size_hidden_layer_1 = len(self.hidden_layer_1_nodes)

        # First, create the second hidden layer
        self.hidden_layer_2_nodes = []
        for node_num in list(range(size_hidden_layer_2)):
            self.hidden_layer_2_nodes.append(Node(node_type='hidden', num_weights=size_hidden_layer_1 + 1,
                                                  node_num=current_node_num, parent_nodes=self.hidden_layer_1_nodes,
                                                  output_class=None))
            current_node_num = current_node_num + 1
        self.depth = self.depth + 1  # Depth should now be 2
        output_nodes_parents = self.hidden_layer_2_nodes
        output_nodes_num_parents = size_hidden_layer_2

        # Update childNodes for hiddenLayer1
        for uniqueNode in self.hidden_layer_1_nodes:
            uniqueNode.childNodes = self.hidden_layer_2_nodes

        # Then, overwrite the output nodes
        if self.dataLoader.dataset in self.dataLoader.categorizationSets:
            output_nodes = []
            for uniquePredictorValue in self.training_data[self.predictor].unique():
                output_nodes.append(Node(node_type='output', num_weights=output_nodes_num_parents + 1,
                                         node_num=current_node_num, parent_nodes=output_nodes_parents,
                                         output_class=uniquePredictorValue))
                current_node_num = current_node_num + 1
        else:
            output_nodes = [Node(node_type='output', num_weights=output_nodes_num_parents + 1,
                                 node_num=current_node_num, parent_nodes=output_nodes_parents, output_class=None)]
        self.output_nodes = output_nodes

        # Update child nodes of hidden layer 2
        for uniqueNode in self.hidden_layer_2_nodes:
            uniqueNode.childNodes = self.output_nodes

    def get_outputs_from_hidden_layer(self, inputs, hidden_layer_num=1):
        """
        Calculates the sigmoid output of each node is a hidden layer for a vector of inputs

        :param inputs: (numeric) A vector of input values into the Nodes
        :param hidden_layer_num: (int) The number of the hidden layer to which to apply the inputs
        :return: (numeric) A vector of outputs from the hidden layer Nodes
        """
        # Create a vector to hold the outputs
        output_vec = []

        # Isolate the nodes that are to produce the outputs
        if hidden_layer_num == 1:
            nodes_to_output = self.hidden_layer_1_nodes
        else:
            nodes_to_output = self.hidden_layer_2_nodes

        for uniqueNode in nodes_to_output:
            # Calculate and append the output from the node
            new_output_val = uniqueNode.calc_sigmoid(inputs=inputs)
            output_vec.append(new_output_val)

            # Update the node's features
            uniqueNode.lastOutputValue = new_output_val
            uniqueNode.lastInputValues = [1] + inputs

        return output_vec

    def get_outputs_from_output_layer(self, inputs):
        """
        Calculates the weighted sum outputs of the Nodes in the output layer

        :param inputs: (numeric) A vector of numeric inputs into the output Nodes
        :return: (numeric) A vector of numeric outputs, the weighted sums of the inputs and the Nodes' weights
        """
        # Create a vector to hold the outputs
        output_vec = []

        for uniqueNode in self.output_nodes:
            # Calculate and append the output from the node
            new_output_val = uniqueNode.calc_weighted_sum_of_inputs(inputs=inputs)
            output_vec.append(new_output_val)

            # Update the node's features
            uniqueNode.lastOutputValue = new_output_val
            uniqueNode.lastInputValues = [1] + inputs

        return output_vec

    def estimate_sample(self, sample):
        """
        Runs a sample through the Network and gets the outputs from the output layer

        :param sample: (DataFrame) A sample to estimate
        :return: (numeric) A vector of outputs from the output layer of the Network
        """
        # First, extract the values of the sample into a vector
        if isinstance(sample, pd.DataFrame):
            remove_cols = ['sampleNum', self.predictor, 'set']
            remove_cols = [col for col in remove_cols if col in sample.columns]
            if len(remove_cols) > 0:
                temp_dt = sample.drop(columns=remove_cols)
            else:
                temp_dt = sample.copy()

            input_vec = temp_dt.values.flatten().tolist()
        else:
            input_vec = sample

        # Then, iterate through each layer. The values of the nodes are updating each time.
        if self.hidden_layer_1_nodes is not None:
            input_vec = self.get_outputs_from_hidden_layer(inputs=input_vec, hidden_layer_num=1)
        if self.hidden_layer_2_nodes is not None:
            input_vec = self.get_outputs_from_hidden_layer(inputs=input_vec, hidden_layer_num=2)

        output_vec = self.get_outputs_from_output_layer(inputs=input_vec)

        return output_vec

    def backpropagate(self, error_vec, learning_rate):
        """
        Update the weights of the Network's Nodes from a vector of errors

        :param error_vec: (numeric) A vector of errors representing the difference of the outputs from their true values
        :param learning_rate: (numeric) The learning rate to apply for weight updates
        :return: None
        """
        # Iterate through each output node. The recursive calls of update_weights will update all the way up the tree
        error_ind = 0
        for outputNode in self.output_nodes:
            output_node_error = error_vec[error_ind]
            outputNode.update_weights(node_error=output_node_error, learning_rate=learning_rate)
            error_ind = error_ind + 1

    def feedforward_and_backpropagate(self, sample, learning_rate):
        """
        Estimates the outputs for a sample, calculates the errors for each output Node, and backpropagates the errors
        to update all the weights in the Network

        :param sample: (DataFrame) A training sample
        :param learning_rate: (int) The learning rate
        :return: None
        """
        # Get the sample's output(s)
        output_vec = self.estimate_sample(sample=sample.copy())

        # Calculate the error(s)
        if self.is_autoencoder:
            remove_cols = ['sampleNum', self.predictor, 'set']
            remove_cols = [col for col in remove_cols if col in sample.columns]
            if len(remove_cols) > 0:
                temp_dt = sample.drop(columns=remove_cols)
            else:
                temp_dt = sample.copy()

            actual_input_values = temp_dt.values.flatten().tolist()

            error_vec = []
            for ind in list(range(len(actual_input_values))):
                error_vec.append(actual_input_values[ind] - output_vec[ind])

        elif self.dataLoader.dataset in self.dataLoader.categorizationSets:
            actual_value = sample[self.predictor].iloc[0]

            # Create a vector of softmax values
            softmax_vals = [math.exp(x) for x in output_vec]
            softmax_total = sum(softmax_vals)
            softmax_vals = [x / softmax_total for x in softmax_vals]

            error_vec = []
            output_index = 0
            for outputNode in self.output_nodes:
                if outputNode.outputClass == actual_value:
                    error_vec.append(1 - softmax_vals[output_index])
                else:
                    error_vec.append(0 - softmax_vals[output_index])
                output_index = output_index + 1
        else:
            actual_value = sample[self.predictor].iloc[0]
            error_vec = [(actual_value - x) for x in output_vec]

        # Backpropagate
        self.backpropagate(error_vec=error_vec, learning_rate=learning_rate)

    def calc_mse(self, testing_set):
        """
        Calculate Mean Squared Error of the Network for a testing dataset

        :param testing_set: (DataFrame) A table of samples to estimate
        :return: (numeric) The MSE of the network
        """
        # Vector to store the predictions for the testing set
        prediction_vec = []

        # Iterate through each testing sample and predict its value
        for sample_ind in list(range(len(testing_set))):
            curr_sample = testing_set.iloc[[sample_ind]]
            new_prediction = self.estimate_sample(sample=curr_sample)
            prediction_vec.append(new_prediction[0])

        comp_dt = testing_set.copy()
        comp_dt['prediction'] = prediction_vec
        comp_dt['squared_error'] = (comp_dt['prediction'] - comp_dt[self.predictor]) ** 2

        mse = sum(comp_dt['squared_error']) / len(comp_dt)

        return mse

    def calc_cross_entropy_loss(self, testing_set):
        """
        Calculates Cross Entropy Loss of the Network for a testing dataset

        :param testing_set: (DataFrame) A table of samples to estimate
        :return: (numeric) The Cross Entropy Loss of the Network
        """
        cel_total = 0

        # Iterate through each sample, estimate its Softmax value for each class
        for sample_ind in list(range(len(testing_set))):
            curr_sample = testing_set.iloc[[sample_ind]].copy()
            new_predictions = self.estimate_sample(sample=curr_sample)
            new_predictions = [math.exp(x) for x in new_predictions]
            predictions_total = sum(new_predictions)
            new_predictions = [x / predictions_total for x in new_predictions]

            # Get the Softmax value for the actual class
            for output_node_ind in list(range(len(self.output_nodes))):
                curr_output_node = self.output_nodes[output_node_ind]
                if curr_output_node.outputClass == curr_sample[self.predictor].iloc[0]:
                    cel_total = cel_total + new_predictions[output_node_ind]

        # Return the negative, since this is a Loss function we want to minimize
        return -1 * cel_total

    def calc_hit_rate(self, testing_set):
        """
        Calculate the classification hit rate of the Network for a testing dataset

        :param testing_set: (DataFrame) A table of samples to classify
        :return: (numeric) The Network's hit rate on the testing set
        """
        prediction_vec = []

        for sample_ind in list(range(len(testing_set))):
            curr_sample = testing_set.iloc[[sample_ind]].copy()
            new_predictions = self.estimate_sample(sample=curr_sample)
            winning_ind = new_predictions.index(max(new_predictions))

            outputClasses = []
            for outputNode in self.output_nodes:
                outputClasses.append(outputNode.outputClass)

            prediction_vec.append(self.output_nodes[winning_ind].outputClass)

        comp_dt = testing_set.copy()
        comp_dt['prediction'] = prediction_vec
        comp_dt['hit'] = (comp_dt[self.predictor] == comp_dt['prediction'])
        comp_dt['hit'] = comp_dt['hit'].astype(int)

        hit_rate = sum(comp_dt['hit']) / len(comp_dt)

        return hit_rate

    def calc_autoencoder_loss(self, testing_set):
        """
        Calculate the loss function for an autoencoder

        :param testing_set: (DataFrame) A table of samples to autoencode
        :return: (numeric) The Loss of the autoencoder Network
        """
        # Variable to sum the total squared errors of the predictions and actual values
        squared_error = 0

        for sample_ind in list(range(len(testing_set))):
            curr_sample = testing_set.iloc[[sample_ind]].copy()
            new_predictions = self.estimate_sample(sample=curr_sample)

            # Convert the sample's features into a list
            remove_cols = ['sampleNum', self.predictor, 'set']
            remove_cols = [col for col in remove_cols if col in curr_sample.columns]
            if len(remove_cols) > 0:
                temp_dt = curr_sample.drop(columns=remove_cols)
            else:
                temp_dt = curr_sample.copy()

            input_vec = temp_dt.values.flatten().tolist()

            for newPredictionInd in list(range(len(new_predictions))):
                squared_error = squared_error + ((new_predictions[newPredictionInd] - input_vec[newPredictionInd]) ** 2)

        # Return the square root of the total squared error
        return math.sqrt(squared_error)

    def calc_loss(self, testing_set):
        """
        Calculate the Loss for the Network

        :param testing_set: (DataFrame) A table of samples for which to calculate Loss
        :return: (numeric) The Loss of the Network
        """
        if self.is_autoencoder:
            output_val = self.calc_autoencoder_loss(testing_set=testing_set)
        elif self.dataLoader.dataset in self.dataLoader.categorizationSets:
            output_val = self.calc_cross_entropy_loss(testing_set=testing_set)
        else:
            output_val = self.calc_mse(testing_set=testing_set)

        return output_val

    def calc_null_model(self, testing_set):
        """
        Calculate the MSE or hit rate of the Null model

        :param testing_set: (DataFrame) A table of samples to classify/regress by the Null model
        :return: (numeric) The MSE or hit rate of the Null model
        """

        if self.is_autoencoder:
            return None

        # If the dataset is categorization, use the most common class value
        if self.dataLoader.dataset in self.dataLoader.categorizationSets:
            null_prediction = testing_set[self.predictor].mode()[0]
            numerator = len(testing_set[testing_set[self.predictor] == null_prediction])
        else:
            null_prediction = testing_set[self.predictor].mean()
            squared_errors = [(x - null_prediction) ** 2 for x in testing_set[self.predictor]]
            numerator = sum(squared_errors)

        return numerator / len(testing_set)



if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

    # Linear network
    data_loader1 = DataLoader(dataset='machine')
    data_loader1.load_saved_data()
    training_set1 = data_loader1.trainingData.copy()
    training_set1 = data_loader1.normalize_data(data=training_set1)
    network1 = Network(data_loader1, is_autoencoder=False, size_hidden_layer_1=None, size_hidden_layer_2=None)
    outputs1 = network1.estimate_sample(sample=training_set1.iloc[[0]].copy())
    network1.feedforward_and_backpropagate(sample=training_set1.iloc[[0]].copy(), learning_rate=.001)
    outputs1_2 = network1.estimate_sample(sample=training_set1.iloc[[0]].copy())
    print("Regression estimate before: ")
    print(outputs1)
    print("Regression estimate after: ")
    print(outputs1_2)
    print()
    print()
    print()
    print()
    print()

    # Linear network
    data_loader1 = DataLoader(dataset='breast-cancer-wisconsin')
    data_loader1.load_saved_data()
    training_set1 = data_loader1.trainingData.copy()
    training_set1 = data_loader1.normalize_data(data=training_set1)
    network1 = Network(data_loader1, is_autoencoder=False, size_hidden_layer_1=None, size_hidden_layer_2=None)
    outputs1 = network1.estimate_sample(sample=training_set1.iloc[[0]].copy())
    network1.feedforward_and_backpropagate(sample=training_set1.iloc[[0]].copy(), learning_rate=.001)
    outputs1_2 = network1.estimate_sample(sample=training_set1.iloc[[0]].copy())
    print("Regression estimate before: ")
    print(outputs1)
    print("Regression estimate after: ")
    print(outputs1_2)
    print()
    print()
    print()
    print()
    print()

    # Multilayer network
    data_loader2 = DataLoader(dataset='machine')
    data_loader2.load_saved_data()
    training_set2 = data_loader2.trainingData.copy()
    training_set2 = data_loader2.normalize_data(data=training_set2)
    network2 = Network(data_loader2, is_autoencoder=False, size_hidden_layer_1=3, size_hidden_layer_2=3)
    outputs2 = network2.estimate_sample(sample=training_set2.iloc[[0]].copy())
    network2.feedforward_and_backpropagate(sample=training_set2.iloc[[0]].copy(), learning_rate=.001)
    outputs2_2 = network2.estimate_sample(sample=training_set2.iloc[[0]].copy())
    print("Regression estimate before: ")
    print(outputs2)
    print("Regression estimate after: ")
    print(outputs2_2)
    print()
    print()
    print()
    print()
    print()

    # Autoencoder network
    data_loader3 = DataLoader(dataset='machine')
    data_loader3.load_saved_data()
    training_set3 = data_loader3.trainingData.copy()
    training_set3 = data_loader3.normalize_data(data=training_set3)
    network3 = Network(data_loader3, is_autoencoder=True, size_hidden_layer_1=4, size_hidden_layer_2=None)
    outputs3 = network3.estimate_sample(sample=training_set3.iloc[[0]].copy())
    network3.feedforward_and_backpropagate(sample=training_set3.iloc[[0]].copy(), learning_rate=.001)
    outputs3_2 = network3.estimate_sample(sample=training_set3.iloc[[0]].copy())
    print()
    print("Reconstruction estimates before: ")
    print(outputs3)
    print("Reconstruction estimates after: ")
    print(outputs3_2)
    print("Actual values: ")
    print(training_set3.iloc[[0]])

