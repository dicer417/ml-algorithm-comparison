import pandas as pd
import os
from src.data_loader import DataLoader
from src.regression_tree import RegressionTree


class Pruner:
    """
    A class to create and prune multiple decision trees for a specific dataset
    """
    def __init__(self, dataset, reprocess_data=True):
        """
        Initializes a new Pruner object for the specified dataset

        :param dataset: (string) Dataset for which to prune the corresponding decision trees
        :param reprocess_data: (logical) Whether to reprocess the data underlying a decision tree,
        for training, pruning, and testing
        """
        self.dataset = dataset
        self.data_loader = DataLoader(dataset=dataset)
        self.categorizationSets = self.data_loader.categorizationSets
        if reprocess_data:
            self.data_loader.load_data()
            self.data_loader.pre_process_data()
            self.data_loader.split_and_save_data()
        else:
            self.data_loader.load_saved_data()

    def create_trees(self, iterations=5):
        """
        Creates iterations x 2 decision trees from the training set, randomly splitting it in half each time,
        with stratification for classification datasets

        :param iterations: (int) Number of times to split the training dataset and train a tree from each half
        """
        save_num = 1
        for ind in list(range(0, iterations)):
            # Randomly split the training data into 2 sets, with stratification
            self.data_loader.split_training_data()

            training_data = self.data_loader.trainingData
            training_set1 = training_data[training_data['set'] == 1].copy()
            training_set2 = training_data[training_data['set'] == 2].copy()

            # Normalize the numeric data
            training_set1 = self.data_loader.normalize_data(training_set1)
            training_set2 = self.data_loader.normalize_data(training_set2)

            # Save the normalized training sets - so we can test later
            file_name_training_set1 = "data/split_training_sets/" + self.dataset + str(save_num) + ".csv"
            training_set1.to_csv(file_name_training_set1, index=False)
            file_name_training_set2 = "data/split_training_sets/" + self.dataset + str(save_num + 1) + ".csv"
            training_set2.to_csv(file_name_training_set2, index=False)

            # Create the tree
            tree = RegressionTree(self.data_loader)

            # First, create a tree from the first training set and save it
            tree.nodes = pd.DataFrame()
            tree.create_tree(data=training_set1)
            file_name = "data/trees/" + self.dataset + str(save_num) + ".csv"
            tree.nodes.to_csv(file_name, index=False)

            save_num = save_num + 1

            # Then, create a tree from the second training set and save it
            tree.nodes = pd.DataFrame()
            tree.create_tree(data=training_set2)
            file_name = "data/trees/" + self.dataset + str(save_num) + ".csv"
            tree.nodes.to_csv(file_name, index=False)

            save_num = save_num + 1

    def prune_tree(self, tree_num=1, write_pruned_tree=False):
        """
        Prunes a saved decision tree

        :param tree_num: (int) The number of the tree to prune
        :param write_pruned_tree: (logical) Whether to save the table of nodes for the newly pruned tree
        :return:
        """
        # Create the tree object and load in the saved nodes
        raw_tree = RegressionTree(self.data_loader)
        raw_tree.load_nodes(num=tree_num, pruned=False)

        # Track the current nodes
        current_nodes = raw_tree.nodes.copy()

        # Get the tuning data and normalize it
        tuning_data = self.data_loader.tuningData
        tuning_data = self.data_loader.normalize_data(tuning_data)

        # Get a list of all of the potential parents to make leaves
        leaf_nodes = current_nodes[current_nodes['isLeaf'] == True]
        curr_parent_node_nums_all = leaf_nodes['parentNodeNum'].unique()
        curr_parent_node_nums_all = [int(x) for x in curr_parent_node_nums_all]
        print(curr_parent_node_nums_all)
        # Check that all the parents only have leaves as children - if not, do not check them for pruning
        curr_parent_node_nums = []
        for uniqueCurrParentNum in curr_parent_node_nums_all:
            children_are_leafs = current_nodes.loc[current_nodes['parentNodeNum'] == uniqueCurrParentNum, 'isLeaf']
            if all(children_are_leafs):
                curr_parent_node_nums.append(uniqueCurrParentNum)

        print(curr_parent_node_nums)

        # Track the nodes that should not be pruned
        nodes_to_keep = []

        # Iterate through each parent in the tree and determine if it can be pruned
        while len(curr_parent_node_nums) > 0:
            # For the current parent node, gather the sub tree
            uniqueParentNodeNum = curr_parent_node_nums[0]
            sub_tree_nodes = current_nodes[((current_nodes['parentNodeNum'] == uniqueParentNodeNum) |
                                            (current_nodes['nodeNum'] == uniqueParentNodeNum))]
            temp_tree = RegressionTree(self.data_loader)
            temp_tree.nodes = sub_tree_nodes

            # Get the tuning data subset that reaches the parentNode
            parent_node_query = sub_tree_nodes.loc[sub_tree_nodes[
                                                       'nodeNum'] == uniqueParentNodeNum, 'fullCondition'].values[0]
            tuning_data_subset = tuning_data.query(parent_node_query)

            # Get the base performance of the tree
            curr_perf = temp_tree.get_tree_hit_rate_or_mse(test_data=tuning_data_subset)
            temp_tree.make_parent_leaf(node_dt=temp_tree.nodes, parent_node_num=uniqueParentNodeNum, update_nodes=True)
            new_perf = temp_tree.get_tree_hit_rate_or_mse(test_data=tuning_data_subset)

            print("Node being evaluated = " + str(uniqueParentNodeNum))
            print("Current subtree performance = " + str(curr_perf))
            print("Pruned subtree performance = " + str(new_perf))

            # Update the current nodes if performance has improved or stayed the same
            if ((self.dataset in self.categorizationSets and new_perf >= curr_perf) or
                    (self.dataset not in self.categorizationSets and new_perf <= curr_perf)):
                current_nodes = raw_tree.make_parent_leaf(node_dt=current_nodes, parent_node_num=uniqueParentNodeNum,
                                                          update_nodes=False)
                print("REMOVED")
            else:
                nodes_to_keep.append(uniqueParentNodeNum)

            # Update our tracker
            leaf_nodes = current_nodes[current_nodes['isLeaf'] == True]
            curr_parent_node_nums_all = leaf_nodes['parentNodeNum'].unique()
            curr_parent_node_nums_all = [int(x) for x in curr_parent_node_nums_all]
            # Check that all the parents only have leaves as children - if not, do not check them for pruning
            curr_parent_node_nums = []
            for uniqueCurrParentNum in curr_parent_node_nums_all:
                children_are_leafs = current_nodes.loc[current_nodes['parentNodeNum'] == uniqueCurrParentNum, 'isLeaf']
                if all(children_are_leafs):
                    curr_parent_node_nums.append(uniqueCurrParentNum)
            # Remove the nodes that have been marked to keep
            curr_parent_node_nums = list(set(curr_parent_node_nums) - set(nodes_to_keep))
            print(curr_parent_node_nums)

            # Halt if the only one node left is the root node
            if len(curr_parent_node_nums) == 1 and curr_parent_node_nums[0] == 1:
                break

        # Write the pruned tree to csv
        if write_pruned_tree:
            pruned_tree_file = "data/trees_pruned/" + self.dataset + str(tree_num) + ".csv"
            current_nodes.to_csv(pruned_tree_file, index=False)

        return current_nodes

    def prune_dataset_trees(self):
        """
        Prunes and saves all 10 trees for a dataset
        :return:
        """
        for tree_ind in list(range(1, 11)):
            self.prune_tree(tree_num=tree_ind, write_pruned_tree=True)

    def test_dataset_pruned_trees(self, write_accuracy_dt=False):
        """
        Tests the performance, as measured by hit rate or MSE, of classifying/regressing the other of the training
        data that was not used to train a given tree

        :param write_accuracy_dt: (logical) Whether to save the resulting table of accuracies in a csv
        :return: (DataFrame) The table of calculated accuracies of the trees for the dataset
        """
        # Create a tree object
        tree = RegressionTree(self.data_loader)

        # Variables to store the accuracies of each tree
        raw_scores = []
        pruned_scores = []
        null_accuracies = []

        # Prepare to get the Null model accuracy
        self.data_loader.load_saved_data()
        if self.dataset in self.categorizationSets:
            null_prediction = self.data_loader.trainingData[self.data_loader.predictor].mode()[0]
        else:
            null_prediction = self.data_loader.trainingData[self.data_loader.predictor].mean()

        # Iterate through each tree, both its raw and pruned version, and get the accuracy
        for tree_ind in list(range(1, 11)):
            print(tree_ind)
            # Get the testing set, the 40% that was used to train a different tree
            test_set_file_name = "data/split_training_sets/" + self.dataset
            if tree_ind % 2 == 0:
                test_set_file_name = test_set_file_name + str(tree_ind - 1)
            else:
                test_set_file_name = test_set_file_name + str(tree_ind + 1)
            test_set_file_name = test_set_file_name + ".csv"

            test_set = pd.read_csv(test_set_file_name)

            # Get the unpruned tree nodes
            tree.load_nodes(num=tree_ind, pruned=False)
            raw_score = tree.get_tree_hit_rate_or_mse(test_data=test_set)
            raw_scores.append(raw_score)

            print("Raw score complete")

            # Get the pruned tree nodes
            tree.load_nodes(num=tree_ind, pruned=True)
            pruned_score = tree.get_tree_hit_rate_or_mse(test_data=test_set)
            pruned_scores.append(pruned_score)

            print("Pruned score complete")

            # Get the NULL model accuracy
            test_set['null_prediction'] = null_prediction
            if self.dataset in self.categorizationSets:
                test_set['null_hit'] = (test_set[self.data_loader.predictor] == test_set['null_prediction'])
                test_set['null_hit'] = test_set['null_hit'].astype(int)
                null_accuracy = sum(test_set['null_hit']) / len(test_set)
            else:
                test_set['null_se'] = (test_set[self.data_loader.predictor] - test_set['null_prediction']) ** 2
                null_accuracy = sum(test_set['null_se']) / len(test_set)
            null_accuracies.append(null_accuracy)

            print("Null score complete")

        # Create a table to output
        accuracy_dt = pd.DataFrame({
            'treeNum': list(range(1, 11)),
            'rawAccuracy': raw_scores,
            'prunedAccuracy': pruned_scores,
            'nullAccuracy': null_accuracies
        })

        # Save the accuracy table for use in the paper
        if write_accuracy_dt:
            acc_file_name = "data/accuracy/" + self.dataset + ".csv"
            accuracy_dt.to_csv(acc_file_name, index=False)

        return accuracy_dt

    def create_full_table(self):
        """
        Loads the saved accuracy table of the dataset and adds columns for the number of nodes in each tree,
        as well as a summary row at the bottom

        :return:
        """
        # Read in accuracy table
        acc_file_name = "data/accuracy/" + self.dataset + ".csv"
        acc_dt = pd.read_csv(acc_file_name)

        # Lists of the number of nodes in each tree
        raw_sizes = []
        pruned_sizes = []

        # Get the sizes of each tree
        for ind in list(range(1, 11)):
            raw_tree_file_name = "data/trees/" + self.dataset + str(ind) + ".csv"
            pruned_tree_file_name = "data/trees_pruned/" + self.dataset + str(ind) + ".csv"

            raw_tree_nodes = pd.read_csv(raw_tree_file_name)
            pruned_tree_nodes = pd.read_csv(pruned_tree_file_name)

            raw_sizes.append(len(raw_tree_nodes))
            pruned_sizes.append(len(pruned_tree_nodes))

        # Create the full table
        full_dt = acc_dt.copy()
        full_dt['rawSize'] = raw_sizes
        full_dt['prunedSize'] = pruned_sizes

        # Create a line for the overall performance
        full_dt['treeNum'] = full_dt['treeNum'].astype(str)

        full_dt_add = pd.DataFrame({
            'treeNum': ['Total'],
            'rawAccuracy': [full_dt['rawAccuracy'].mean()],
            'prunedAccuracy': [full_dt['prunedAccuracy'].mean()],
            'nullAccuracy': [full_dt['nullAccuracy'].mean()],
            'rawSize': [full_dt['rawSize'].mean()],
            'prunedSize': [full_dt['prunedSize'].mean()]
        })

        full_dt = pd.concat([full_dt, full_dt_add])

        # Force datatypes
        full_dt['rawSize'] = full_dt['rawSize'].astype(int)
        full_dt['prunedSize'] = full_dt['prunedSize'].astype(int)

        # Reorder columns
        new_col_ord = ['treeNum', 'rawSize', 'prunedSize', 'rawAccuracy', 'prunedAccuracy', 'nullAccuracy']
        full_dt = full_dt[new_col_ord]

        # Rename columns
        if self.dataset in self.categorizationSets:
            new_col_names = ['Tree', 'Raw Tree Size', 'Pruned Tree Size',
                             'Raw Hit Rate', 'Pruned Hit Rate', 'Null Model']
        else:
            new_col_names = ['Tree', 'Raw Tree Size', 'Pruned Tree Size',
                             'Raw MSE', 'Pruned MSE', 'Null Model']

        full_dt.columns = new_col_names

        return full_dt


if __name__ == '__main__':
    os.chdir('C:\\Users\\toddi\\PycharmProjects\\Programming-Project-2')

    pruner = Pruner('house-votes-84', reprocess_data=False)
    pruner.prune_tree(tree_num=1, write_pruned_tree=False)

    # pruner = Pruner('house-votes-84', reprocess_data=False)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     curr_nodes = pruner.prune_tree(tree_num=1)
    #     print(curr_nodes)
    # pruner.create_trees(iterations=5)

    # valid_datasets = ['abalone', 'breast-cancer-wisconsin', 'car', 'forestfires', 'house-votes-84', 'machine']
    # # valid_datasets = ['machine']
    # for uniqueDataset in valid_datasets:
    #     pruner = Pruner(dataset=uniqueDataset, reprocess_data=False)
    #     full_dt = pruner.create_full_table()
    #     print(uniqueDataset)
    #     print(full_dt.to_latex(index=False))
    #     print()
    #     print()
    #     # acc_dt = pruner.test_dataset_pruned_trees(write_accuracy_dt=True)
    # #     pruner.create_trees(iterations=5)
    # #     pruner.prune_dataset_trees()




