import math
import pandas as pd
import os
from src.data_loader import DataLoader


class RegressionTree:
    """
    A class to create a regression tree from the data in a DataLoader object
    """
    def __init__(self, data_loader):
        """
        Initializes a RegressionTree object from a DataLoader object

        :param data_loader: A DataLoader object for the desired dataset
        """
        self.data_loader = data_loader              # Stores the underlying data for the tree
        self.dataset = data_loader.dataset          # Name of the dataset
        self.predictor = data_loader.predictor      # Value to be predicted (i.e. classified or regressed)
        self.rootNodeNum = 0                        # Node number for the root node
        self.nodes = pd.DataFrame()                 # Table containing the nodes of the tree

    def calc_categorical_feature_gain(self, data, feature):
        """
        Calculates the gain of using a specific categorical feature as the splitting criterion for a dataset

        :param data: (DataFrame) The dataset that has reached the node to be defined
        :param feature: (string) Name of the column to use as the splitting criterion
        :return: (int) The gain
        """
        total_n = len(data)
        # Calculate current entropy
        current_entropy = 0
        for uniqueClass in data[self.predictor].unique():
            entropy_add = len(data[data[self.predictor] == uniqueClass]) / total_n
            current_entropy = current_entropy + (entropy_add * math.log2(entropy_add))

        # Calculate expected entropy
        expected_entropy = 0
        for uniqueFeatureVal in data[feature].unique():
            data_feature = data[data[feature] == uniqueFeatureVal]

            # Calculate entropy of the subset of data for which the feature has the specified value
            feature_entropy = 0
            for uniqueClass in data_feature[self.predictor].unique():
                feature_entropy_add = len(data_feature[data_feature[self.predictor] == uniqueClass]) / len(data_feature)
                feature_entropy = feature_entropy + (feature_entropy_add * math.log2(feature_entropy_add))

            expected_entropy = expected_entropy + (len(data_feature) * -1 * feature_entropy / total_n)

        # Calculate gain
        gain = (-1 * current_entropy) - expected_entropy

        return gain

    def calc_categorical_feature_gain_ratio(self, data, feature):
        """
        Calculates the gain ratio of using a specific categorical feature as the splitting criterion for a dataset

        :param data: (DataFrame) The dataset that has reached the node to be defined
        :param feature: (string) Name of the column to use as the splitting criterion
        :return: (int) The gain ratio
        """
        # Calculate gain(fi)
        gain = self.calc_categorical_feature_gain(data=data,feature=feature)

        # Calculate IV(fi)
        feature_iv = 0
        for uniqueFeatureVal in data[feature].unique():
            feature_iv_add = len(data[data[feature] == uniqueFeatureVal]) / len(data)
            feature_iv = feature_iv + (feature_iv_add * math.log2(feature_iv_add))

        return gain / (-1 * feature_iv)

    def calc_numerical_feature_gain(self, data, feature, cutoff):
        """
        Calculates the gain of using a specific numerical feature with a single specific cutoff value
         as the splitting criterion for a dataset

        :param data: (DataFrame) The dataset that has reached the node to be defined
        :param feature: (string) Name of the column to use as the splitting criterion
        :param cutoff: (int) Value to use as the feature's cutoff
        :return: (int) The gain
        """
        total_n = len(data)
        # Calculate current entropy
        current_entropy = 0
        for uniqueClass in data[self.predictor].unique():
            entropy_add = len(data[data[self.predictor] == uniqueClass]) / total_n
            current_entropy = current_entropy + (entropy_add * math.log2(entropy_add))

        # Calculate expected entropy for the data partition greater than the cutoff value
        expected_entropy_greater = 0
        data_greater = data[data[feature] >= cutoff]
        for uniqueClass in data_greater[self.predictor].unique():
            expected_entropy_greater_add = (len(data_greater[data_greater[self.predictor] == uniqueClass]) /
                                            len(data_greater))
            expected_entropy_greater = (expected_entropy_greater +
                                        (expected_entropy_greater_add * math.log1p(expected_entropy_greater_add)))

        # Calculate expected entropy for the data partition less than the cutoff value
        expected_entropy_lesser = 0
        data_lesser = data[data[feature] >= cutoff]
        for uniqueClass in data_lesser[self.predictor].unique():
            expected_entropy_lesser_add = (len(data_lesser[data_lesser[self.predictor] == uniqueClass]) /
                                            len(data_lesser))
            expected_entropy_lesser = (expected_entropy_lesser +
                                       (expected_entropy_lesser_add * math.log1p(expected_entropy_lesser_add)))

        expected_entropy = ((len(data_greater) / len(data)) * -1 * expected_entropy_greater +
                            (len(data_lesser) / len(data)) * -1 * expected_entropy_lesser)

        # Calculate gain
        gain = -1 * current_entropy - expected_entropy

        return gain

    def calc_numerical_feature_gain_ratio(self, data, feature, cutoff):
        """
        Calculates the gain ratio of using a specific numerical feature with a single specific cutoff value
         as the splitting criterion for a dataset

        :param data: (DataFrame) The dataset that has reached the node to be defined
        :param feature: (string) Name of the column to use as the splitting criterion
        :param cutoff: (int) Value to use as the feature's cutoff
        :return: (int) The gain ratio
        """
        # Calculate gain(fi)
        gain = self.calc_numerical_feature_gain(data=data, feature=feature, cutoff=cutoff)

        # Calculate IV(fi)
        feature_iv_greater = len(data[data[feature] >= cutoff]) / len(data)
        feature_iv_lesser = len(data[data[feature] < cutoff]) / len(data)
        feature_iv = -1 * (feature_iv_greater * math.log2(feature_iv_greater) +
                           feature_iv_lesser * math.log2(feature_iv_lesser))

        return gain / feature_iv

    def calc_categorical_feature_squared_error(self, data, feature):
        """
        Calculates the squared error of using a specific categorical feature as the splitting criterion for a dataset

        :param data: (DataFrame) The dataset that has reached the node to be defined
        :param feature: (string) Name of the column to use as the splitting criterion
        :return: (int) The squared error
        """
        squared_error = 0

        for uniqueFeatureVal in data[feature].unique():
            data_branch = data[data[feature] == uniqueFeatureVal]
            # Assume the predicted response value is the mean value of all samples that follow that branch
            predicted_response = data_branch[self.predictor].mean()

            # Calculate the squared error for taking that branch
            branch_squared_error = 0
            for uniqueSampleInd in list(range(0, len(data_branch))):
                branch_squared_error_add = (data_branch[self.predictor].iloc[uniqueSampleInd] - predicted_response) ** 2
                branch_squared_error = branch_squared_error + branch_squared_error_add

            squared_error = squared_error + branch_squared_error

        return squared_error / len(data)

    def calc_numerical_feature_squared_error(self, data, feature, cutoff):
        """
        Calculates the squared error of using a specific numerical feature with a single specific cutoff value
         as the splitting criterion for a dataset

        :param data: (DataFrame) The dataset that has reached the node to be defined
        :param feature: (string) Name of the column to use as the splitting criterion
        :param cutoff: (int) Value to use as the feature's cutoff
        :return: (int) The squared error
        """
        data_greater = data[data[feature] >= cutoff]
        # Assume the predicted response value is the mean value of all samples that follow that branch
        predicted_value_greater = data_greater[self.predictor].mean()

        # Calculate squared error for samples with feature values >= cutoff
        greater_squared_error = 0
        # There may be instances when there are no samples in the greater-than category,
        # perhaps because they all have the same value
        if len(data_greater) > 0:
            for uniqueGreaterInd in list(range(0, len(data_greater))):
                greater_squared_error_add = (data_greater[self.predictor].iloc[uniqueGreaterInd] -
                                             predicted_value_greater) ** 2
                greater_squared_error = greater_squared_error + greater_squared_error_add

        data_lesser = data[data[feature] < cutoff]
        # Assume the predicted response value is the mean value of all samples that follow that branch
        predicted_value_lesser = data_lesser[self.predictor].mean()

        # Calculate squared error for samples with feature values < cutoff
        lesser_squared_error = 0
        # There may be instances when there are no samples in the less-than category, perhaps because they all have the
        # same value
        if len(data_lesser) > 0:
            for uniqueLesserInd in list(range(0, len(data_lesser))):
                lesser_squared_error_add = (data_lesser[self.predictor].iloc[uniqueLesserInd] -
                                            predicted_value_lesser) ** 2
                lesser_squared_error = lesser_squared_error + lesser_squared_error_add

        squared_error = greater_squared_error + lesser_squared_error

        if len(data) == 0:
            print(data)
            print("FEATURE: " + feature)
            # print("CUTOFF: " + cutoff)
            return math.inf

        return squared_error / len(data)

    def select_split_feature(self, data, used_features=None):
        """
        Iterates through all the features in a dataset besides those marked as used and determines which feature
        has the highest gain ratio or lowest MSe

        :param data: (DataFrame) The dataset that has reached the node to be defined
        :param used_features: (list of strings) Features to not consider as splitting criteria
        :return: [the selected feature, the feature's cutoff (None if categorical feature]
        """
        # Determine potential features
        potential_features = self.data_loader.attributes.copy()
        if used_features is not None:
            potential_features = list(set(potential_features) - set(used_features))

        # Remove features for which there is only 1 value
        features_with_one_value = []
        for potentialFeature in potential_features:
            # if (potentialFeature in self.data_loader.nominalAttributes or
            #         potentialFeature in self.data_loader.ordinalAttributes):
            #     if len(data[potentialFeature].unique()) < 2:
            #         features_with_one_value.append(potentialFeature)
            if len(data[potentialFeature].unique()) < 2:
                features_with_one_value.append(potentialFeature)
        potential_features = list(set(potential_features) - set(features_with_one_value))

        # If there are no more potential features, return None
        if len(potential_features) == 0:
            return [None, None]

        # Iterate through each feature and calculate the gain ratios / squared errors
        feature_scores = []
        feature_cutoffs = []
        for uniqueFeature in potential_features:
            if self.dataset in self.data_loader.categorizationSets:
                if uniqueFeature in self.data_loader.numericAttributes:
                    cutoff = data[uniqueFeature].mean()

                    if math.isnan(cutoff):
                        print(data[uniqueFeature])

                    feature_score = self.calc_numerical_feature_gain_ratio(
                        data=data, feature=uniqueFeature, cutoff=cutoff)
                    feature_cutoffs.append(cutoff)
                else:
                    feature_score = self.calc_categorical_feature_gain_ratio(data=data, feature=uniqueFeature)
                    feature_cutoffs.append(None)
            else:
                if uniqueFeature in self.data_loader.numericAttributes:
                    cutoff = data[uniqueFeature].mean()

                    if math.isnan(cutoff):
                        print(data[uniqueFeature])

                    feature_score = self.calc_numerical_feature_squared_error(
                        data=data, feature=uniqueFeature, cutoff=cutoff)

                    if math.isinf(feature_score):
                        print(data)
                        print(used_features)
                        print(uniqueFeature)

                    feature_cutoffs.append(cutoff)
                else:
                    feature_score = self.calc_categorical_feature_squared_error(data=data, feature=uniqueFeature)
                    feature_cutoffs.append(None)

            feature_scores.append(feature_score)

        feature_table = pd.DataFrame({
            'feature': potential_features,
            'feature_score': feature_scores,
            'feature_cutoff': feature_cutoffs
        })
        print(feature_table)

        # Get index of the largest gain ratio / the smallest squared error
        if self.dataset in self.data_loader.categorizationSets:
            feature_ind = feature_scores.index(max(feature_scores))
        else:
            feature_ind = feature_scores.index(min(feature_scores))

        # Finally, return that feature
        selected_split_feature = potential_features[feature_ind]
        selected_split_cutoff = feature_cutoffs[feature_ind]

        return [selected_split_feature, selected_split_cutoff]

    def generate_next_node(self, data=None, parent_node_num=None, used_features=None, full_condition=None):
        """


        :param data: (DataFrame) The dataset that has reached the node to be defined
        :param parent_node_num: (int) Node number of the new node's parent
        :param used_features:  (list of strings) Features to not consider as splitting criteria
        :param full_condition: The condition to reduce the dataset to its subset that reaches the node
        :return: (DataFrame) A table with all of the new node's information
        """
        # Determine if parent
        is_root = parent_node_num is None

        # Determine if leaf
        is_leaf = ((used_features is not None and len(used_features) == len(self.data_loader.attributes))
                   or len(data) == 1)

        # Determine nodeNum, depth
        if is_root:
            nodeNum = 1
            parent_node = None
            depth = 1
        else:
            nodeNum = max(self.nodes['nodeNum']) + 1
            parent_node = self.nodes[self.nodes['nodeNum'] == parent_node_num]
            depth = parent_node['depth'].iloc[0] + 1

        # Determine the split feature boundary if it's not a leaf
        if not is_leaf:
            split_feature_outputs = self.select_split_feature(data=data, used_features=used_features)

            # If no split feature was returned, it's a leaf
            if split_feature_outputs[0] is None:
                is_leaf = True

        # Determine the other node values
        if is_leaf:
            split_feature = None
            split_boundary = None
            if self.dataset in self.data_loader.categorizationSets:
                predicted_value = data[self.predictor].mode()[0]
            else:
                predicted_value = data[self.predictor].mean()
        else:
            split_feature = split_feature_outputs[0]
            if used_features is None:
                used_features = [split_feature]
            else:
                used_features = used_features.copy()  # To not affect the parent's values, deep copy
                used_features.append(split_feature)
            split_boundary = split_feature_outputs[1]
            # predicted_value = None
            if self.dataset in self.data_loader.categorizationSets:
                predicted_value = data[self.predictor].mode()[0]
            else:
                predicted_value = data[self.predictor].mean()

        # Create node DataFrame
        node_df = pd.DataFrame({
            'nodeNum': nodeNum,
            'parentNodeNum': parent_node_num,
            'splitFeature': split_feature,
            'splitBoundary': split_boundary,
            'predictedValue': predicted_value,
            'depth': depth,
            'childNodeDirectory': None,
            'isRoot': is_root,
            'isLeaf': is_leaf,
            'fullCondition': full_condition,
            'used_features': [used_features]
        })

        # Output a DataFrame
        return node_df

    def create_tree(self, data):
        """
        Creates a full decision tree for the provided dataset, with no early stopping

        :param data: (DataFrame) Dataset for which to create a new tree
        """
        # Firstly, create the root node
        root_node = self.generate_next_node(data=data, parent_node_num=None, used_features=None, full_condition=None)
        self.nodes = root_node

        # Keep checking for non-leaf nodes that do not have children
        nodes_to_address = self.nodes[(self.nodes['isLeaf'] == False) & (self.nodes['childNodeDirectory'].isnull())]

        stop_counter = 0
        while len(nodes_to_address) > 0 and stop_counter < 2:
            # # FOR TESTING
            # stop_counter += 1

            # Extract the first node that needs children
            node_to_give_children = nodes_to_address.iloc[[0]]

            # Subset the data to those samples that reach that parent node
            parent_condition = node_to_give_children['fullCondition'].iloc[0]
            if parent_condition is None:
                parent_data = data
            else:
                parent_data = data.query(parent_condition)

            # Extract the parent's information
            parent_feature = node_to_give_children['splitFeature'].iloc[0]
            parent_node_num = node_to_give_children['nodeNum'].iloc[0]
            parent_row_num = self.nodes.index[self.nodes['nodeNum'] == parent_node_num].tolist()
            parent_row_num = parent_row_num[0]
            parent_used_features = node_to_give_children['used_features'].iloc[0]
            if parent_used_features is None:
                child_used_features = [parent_feature]
            else:
                if not isinstance(parent_used_features, list):
                    parent_used_features = [parent_used_features]
                child_used_features = parent_used_features.copy()

            # Only 2 children if the attribute is numeric
            if parent_feature in self.data_loader.numericAttributes:
                parent_cutoff = node_to_give_children['splitBoundary'].iloc[0]

                if math.isnan(parent_cutoff):
                    print("CHECK HERE")
                    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                        print(node_to_give_children)
                    print(parent_condition)

                # First, address the samples with feature values below the boundary
                # Create the child condition
                if parent_condition is None:
                    child_condition1 = "`" + parent_feature + "`<" + str(parent_cutoff)
                else:
                    child_condition1 = parent_condition + " & `" + parent_feature + "`<" + str(parent_cutoff)

                # Generate the next node
                child_data = data.query(child_condition1)
                child_node1 = self.generate_next_node(data=child_data,
                                                      parent_node_num=parent_node_num,
                                                      used_features=child_used_features,
                                                      full_condition=child_condition1)
                # Append the child node to self.nodes
                self.nodes = pd.concat([self.nodes, child_node1], axis=0, ignore_index=True)

                # Add the child node to the parent's dictionary of child nodes
                if self.nodes.at[parent_row_num, 'childNodeDirectory'] is None:
                    self.nodes.at[parent_row_num, 'childNodeDirectory'] = \
                        {'<': child_node1['nodeNum'].iloc[0]}
                else:
                    self.nodes.at[parent_row_num, 'childNodeDirectory']['<'] = (
                        child_node1['nodeNum'].iloc[0])

                # Second, address samples with feature values greater than or equal to the boundary
                # Create the child condition
                if parent_condition is None:
                    child_condition2 = "`" + parent_feature + "`>=" + str(parent_cutoff)
                else:
                    child_condition2 = parent_condition + " & `" + parent_feature + "`>=" + str(parent_cutoff)

                # Generate the next node
                child_data = data.query(child_condition2)
                child_node2 = self.generate_next_node(data=child_data,
                                                      parent_node_num=parent_node_num,
                                                      used_features=child_used_features,
                                                      full_condition=child_condition2)
                # Append the child node to self.nodes
                self.nodes = pd.concat([self.nodes, child_node2], axis=0, ignore_index=True)

                # Add the child node to the parent's dictionary of child nodes
                if self.nodes.at[parent_row_num, 'childNodeDirectory'] is None:
                    self.nodes.at[parent_row_num, 'childNodeDirectory'] = \
                        {'>=': child_node2['nodeNum'].iloc[0]}
                else:
                    self.nodes.at[parent_row_num, 'childNodeDirectory']['>='] = (
                        child_node2['nodeNum'].iloc[0])

            # If categorical/nominal, there are as many children as there are values in the class
            else:
                child_values = parent_data[parent_feature].unique()
                for uniqueChildValue in child_values:
                    # Prepare the addition to the data subsetting conditions
                    if isinstance(uniqueChildValue, str):
                        uniqueChildValue_string = "'" + uniqueChildValue + "'"
                    else:
                        uniqueChildValue_string = str(uniqueChildValue)

                    # Append the condition
                    if parent_condition is None:
                        child_condition = "`" + parent_feature + "`==" + uniqueChildValue_string
                    else:
                        child_condition = parent_condition + " & `" + parent_feature + "`==" + uniqueChildValue_string

                    # Subset the data and use it to calculate the split at each child node
                    child_data = data.query(child_condition)
                    child_node = self.generate_next_node(data=child_data,
                                                         parent_node_num=parent_node_num,
                                                         used_features=child_used_features,
                                                         full_condition=child_condition)

                    # Append the child node to self.nodes
                    self.nodes = pd.concat([self.nodes, child_node], axis=0, ignore_index=True)

                    # Add the child node to the parent's dictionary of child nodes
                    if self.nodes.at[parent_row_num, 'childNodeDirectory'] is None:
                        self.nodes.at[parent_row_num, 'childNodeDirectory'] = \
                            {uniqueChildValue: child_node['nodeNum'].iloc[0]}
                    else:
                        self.nodes.at[parent_row_num, 'childNodeDirectory'][uniqueChildValue] = (
                            child_node['nodeNum'].iloc[0])

            # Update nodes_to_address
            nodes_to_address = self.nodes[(self.nodes['isLeaf'] == False) & (self.nodes['childNodeDirectory'].isnull())]

    def load_nodes(self, num=1, pruned=False):
        """
        Loads a set of pre-saved nodes for a tree already created

        :param num: (int) Number of the tree to load
        :param pruned: (logical) Whether to load the pruned version or the raw version (False)
        """
        # Create file name
        file_name = "data/"
        if pruned:
            file_name = file_name + "trees_pruned/"
        else:
            file_name = file_name + "trees/"
        file_name = file_name + self.dataset + str(num) + ".csv"
        new_nodes = pd.read_csv(file_name)

        self.nodes = new_nodes

    def predict_sample(self, sample):
        """
        Predicts the class or regression value for a provided sample using the tree saved in self.nodes

        :param sample: (DataFrame) A single row of a dataset to classify
        :return: The classification (string) or regressed value (numeric)
        """
        predicted_value = None

        # Iterate through each leaf node in the set
        # Once a leaf's condition is found for which the sample fits, return that leaf's predicted value
        leaf_nodes = self.nodes[self.nodes['isLeaf'] == True]
        for index, row in leaf_nodes.iterrows():
            leaf_condition = row["fullCondition"]
            leaf_sample = sample.query(leaf_condition)

            if len(leaf_sample) > 0:
                predicted_value = row["predictedValue"]

                print("Leaf node chosen:")
                print("Node #" + str(row["nodeNum"]))
                print("Node condition " + row['fullCondition'])

                break

        # If no leaf is reached, return the Null model value - the predicted value at the root node
        if predicted_value is None:
            predicted_value = self.nodes['predictedValue'].iloc[0]

        return predicted_value

    def get_tree_hit_rate_or_mse(self, test_data):
        """
        Classifies or regresses each sample provided, compares those predictions against the samples' true values,
        and calculates the hit rate or MSE of its predictions

        :param test_data: (DataFrame) The samples to classify/regress
        :return: (numeric) Hit rate or MSE
        """
        # Return 0 if there's no test data
        if len(test_data) == 0:
            return 0

        # Copy test data as we will manipulate it to calculate hit rate / MSE
        test_data_temp = test_data.copy()

        # Store the predicted values in a list
        predictions = []

        # Iterate through each sample in the test dataset and get a prediction
        for rowInd in list(range(0, len(test_data_temp))):
            print("Test sample #" + str(rowInd))
            predict_sample = test_data_temp.iloc[[rowInd]]
            predicted_value = self.predict_sample(sample=predict_sample)
            predictions.append(predicted_value)

        test_data_temp['prediction'] = predictions

        # Calculate hit rate
        if self.dataset in self.data_loader.categorizationSets:
            test_data_temp['hit'] = (test_data_temp[self.predictor] == test_data_temp['prediction'])
            test_data_temp['hit'] = test_data_temp['hit'].astype(int)

            return_val = sum(test_data_temp['hit']) / len(test_data_temp)
        else:
            test_data_temp['squared_error'] = (test_data_temp[self.predictor] - test_data_temp['prediction']) ** 2
            return_val = sum(test_data_temp['squared_error']) / len(test_data_temp)

        return return_val

    def make_parent_leaf(self, node_dt, parent_node_num, update_nodes=False):
        """
        For use in pruning a tree - removes a parent's children from a tree and sets it as a leaf

        :param node_dt: (DataFrame) A table of a tree's nodes
        :param parent_node_num: (int) The number of the parent node to make a leaf
        :param update_nodes: (logical) Whether to update the self.nodes attribute of the RegressionTree object
        :return:
        """
        new_node_dt = node_dt.copy()

        # Remove all the parent's children
        new_node_dt = new_node_dt[new_node_dt['parentNodeNum'] != parent_node_num]

        # Adjust the parent's values
        new_node_dt.loc[new_node_dt['nodeNum'] == parent_node_num, 'isLeaf'] = True
        new_node_dt.loc[new_node_dt['nodeNum'] == parent_node_num, 'childNodeDictionary'] = None

        if update_nodes:
            self.nodes = new_node_dt

        return new_node_dt

# nodeNum   parentNodeNum  splitFeature  splitBoundary   predictedValue   depth   childNodeDirectory
# isRoot  isLeaf    fullCondition   used_features


if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

    data_loader1 = DataLoader('breast-cancer-wisconsin')
    data_loader1.load_saved_data()
    tree1 = RegressionTree(data_loader=data_loader1)
    dataset1 = data_loader1.trainingData.copy()
    dataset1 = data_loader1.normalize_data(dataset1)

    # Feature gain ----
    gain = tree1.calc_numerical_feature_gain(data=dataset1, feature="Mitoses",
                                             cutoff=dataset1['Clump Thickness'].mean())
    print("Gain from Clump Thickness:")
    print(gain)
    print()

    # Feature gain ratio ----
    gain_ratio = tree1.calc_numerical_feature_gain_ratio(data=dataset1, feature="Mitoses",
                                                         cutoff=dataset1['Clump Thickness'].mean())
    print("Gain ratio from Clump Thickness:")
    print(gain_ratio)
    print()

    # Feature selection and node creation ----
    example_node = tree1.generate_next_node(data=dataset1, parent_node_num=None, used_features=None,
                                            full_condition=None)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print()
        print("Root node of a fresh tree for Breast Cancer Wisconsin:")
        print(example_node)
        print()
        print("------------------------------------------------------------------------------------------------------")
        print()

    # Mean squared error
    data_loader2 = DataLoader('machine')
    data_loader2.load_saved_data()
    tree2 = RegressionTree(data_loader=data_loader2)
    dataset2 = data_loader2.trainingData.copy()
    dataset2 = data_loader2.normalize_data(dataset2)
    example_mse = (
        tree2.calc_numerical_feature_squared_error(data=dataset2, feature='MMAX', cutoff=dataset2['MMAX'].mean()))
    print("MSE of MMAX:")
    print(example_mse)
    print()

    example_node2 = tree2.generate_next_node(data=dataset2, parent_node_num=None, used_features=None,
                                             full_condition=None)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print()
        print("Root node of a fresh tree for Machine:")
        print(example_node2)
        print()
        print("------------------------------------------------------------------------------------------------------")
        print()

    # Classify an example
    test_set1 = data_loader1.tuningData.copy()
    test_set1 = data_loader1.normalize_data(data=test_set1)
    tree1.load_nodes(num=1, pruned=True)
    classification = tree1.predict_sample(sample=test_set1.iloc[[0]].copy())
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(test_set1.iloc[[0]])
        print("Predicted class: " + str(classification))
        print()
        print("------------------------------------------------------------------------------------------------------")
        print()

    # Regress an example
    test_set2 = data_loader2.tuningData.copy()
    test_set2 = data_loader2.normalize_data(data=test_set2)
    tree2.load_nodes(num=1, pruned=True)
    regression = tree2.predict_sample(sample=test_set2.iloc[[0]].copy())
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(test_set2.iloc[[0]])
        print("Predicted value: " + str(regression))
        print()
        print("------------------------------------------------------------------------------------------------------")
        print()


    # Display trees
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("Pruned Breast Cancer Wisconsin tree #1")
        print(tree1.nodes)
        print()
        print()
        print("Pruned Machine tree #1")
        print(tree2.nodes)
        print()
        print()

