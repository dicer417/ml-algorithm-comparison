import os
import pandas as pd
import jinja2

from src.DataLoader import DataLoader
from src.Network import Network


class NetworkTuner:
    """
    A Class to run the tuning and testing of Networks for different parameters
    """
    def __init__(self, dataset, reload_data=False):
        """
        Initializes a new NetworkTuner object

        :param dataset: (String) Name of the dataset for which to tune
        :param reload_data: (Boolean) Whether to reprocess the raw data
        """
        self.dataset = dataset

        # Load and/or process the data
        data_loader = DataLoader(dataset=dataset)
        if reload_data:
            data_loader.load_data()
            data_loader.assign_headers()
            data_loader.pre_process_data()
            data_loader.split_and_save_data()
        else:
            data_loader.load_saved_data()

        self.dataLoader = data_loader

    def train_network(self, network, training_set, tuning_set, learning_rate, max_iterations, check_increment):
        """
        Trains a Network using the parameters provided via feedforward and backpropagation

        :param network: (Network) The Network to train
        :param training_set: (DataFrame) A table of samples to feedforward and backpropagate
        :param tuning_set: (DataFrame) A table of samples with which to periodically calculate Loss
        :param learning_rate: (numeric) Learning rate by which to train
        :param max_iterations: (int) Maximum number of training samples to apply --> ensures the training doesn't
        run too long
        :param check_increment: (int) Number of training samples after which to check if the Loss is still improving
        :return: None
        """
        # Track the current Loss of the Network
        current_performance = None

        # Control variables for the while loop
        curr_training_ind = 0
        continue_loop = True
        num_iterations = 0

        # Continually feedforward and backpropagate samples until a stopping criterion is met
        while continue_loop and num_iterations < max_iterations:
            # Feedforward and backpropagate a training sample
            training_sample = training_set.iloc[[curr_training_ind]].copy()
            network.feedforward_and_backpropagate(sample=training_sample, learning_rate=learning_rate)

            # Increment our trackers forward
            curr_training_ind = curr_training_ind + 1
            num_iterations = num_iterations + 1

            # Check performance every _ iterations. If it does not improve by at least 1%, end the training
            if num_iterations % check_increment == 0:
                new_performance = network.calc_loss(testing_set=tuning_set)

                if current_performance is not None:
                    # Check if performance has improved at least 1%. If not, end training.
                    improvement = abs((current_performance - new_performance) / current_performance)
                    if new_performance < current_performance and improvement > .01:
                        current_performance = new_performance
                    else:
                        continue_loop = False
                else:
                    current_performance = new_performance

            # If we've reached the end of the training set, shuffle it and start again
            if curr_training_ind >= len(training_set):
                curr_training_ind = 0
                training_set = training_set.sample(frac=1).reset_index(drop=True)

        return num_iterations

    def tune_for_a_dataset(self, network_type, iterations=5, learning_rates=None, hidden_layer_sizes1=None,
                           hidden_layer_sizes2=None):
        """
        Performs _x2 cross validation for each combination of inputs provided

        :param network_type: (String) 'autoencoder', 'simple', or 'layered'
        :param iterations: (int) Number of cross validations to perform
        :param learning_rates: (numeric) Learning rates to test
        :param hidden_layer_sizes1: (int) Sizes of the first hidden layer to test
        :param hidden_layer_sizes2: (int) Sizes of the second hidden layer to test
        :return: None
        """

        # Default learning rate
        if learning_rates is None:
            learning_rates = [.01]

        # Create a table of the combinations of inputs to use
        input_combos = []
        for uniqueLearningRate in learning_rates:
            if hidden_layer_sizes1 is not None:
                for uniqueHiddenLayerSize1 in hidden_layer_sizes1:
                    if hidden_layer_sizes2 is not None:
                        for uniqueHiddenLayerSize2 in hidden_layer_sizes2:
                            input_combos.append([uniqueLearningRate, uniqueHiddenLayerSize1, uniqueHiddenLayerSize2])
                    else:
                        input_combos.append([uniqueLearningRate, uniqueHiddenLayerSize1, None])
            else:
                input_combos.append([uniqueLearningRate, None, None])

        input_combos = pd.DataFrame(input_combos)

        # Assign column names to the input combos
        input_combo_names = ['learningRate', 'hiddenLayerSize1', 'hiddenLayerSize2']
        input_combo_names = input_combo_names[0:len(input_combos.columns)]
        input_combos.columns = input_combo_names

        # For each iteration, split the training set and normalize them.
        # Then, train a model for each combination of inputs.
        for iteration_ind in list(range(iterations * 2)):
            if iteration_ind % 2 == 0:
                train_set_num = 1
                test_set_num = 2
                # Only re-split the training data every odd run
                self.dataLoader.split_training_data()
            else:
                train_set_num = 2
                test_set_num = 1

            # Set datasets
            training_set = self.dataLoader.trainingData[self.dataLoader.trainingData["set"] == train_set_num].copy()
            testing_set = self.dataLoader.trainingData[self.dataLoader.trainingData["set"] == test_set_num].copy()
            tuning_set = self.dataLoader.tuningData.copy()

            # Normalize datasets
            training_set = self.dataLoader.normalize_data(data=training_set)
            testing_set = self.dataLoader.normalize_data(data=testing_set)
            tuning_set = self.dataLoader.normalize_data(data=tuning_set)

            # Set autoencoder parameter
            is_autoencoder = (type == 'autoencoder')

            # For each combination of inputs, train a network and test it
            for inputInd in list(range(len(input_combos))):
                # Extract network training parameters, i.e. the parameters to tune
                learning_rate = input_combos["learningRate"].iloc[inputInd]
                size_hidden_layer_1 = input_combos["hiddenLayerSize1"].iloc[inputInd]
                size_hidden_layer_2 = input_combos["hiddenLayerSize2"].iloc[inputInd]

                print("Dataset=" + self.dataset + "; Iteration=" + str(iteration_ind + 1) +
                      "; Learning Rate=" + str(learning_rate) + "; Hidden Layer 1 Size=" +
                      str(size_hidden_layer_1) + "; Hidden Layer 2 Size=" + str(size_hidden_layer_2))

                # Set max iterations and check increments --> to ensure it doesn't run too long or too short
                max_iterations = len(training_set) * 200
                check_increment = len(training_set) * 2

                # Create a new network and train it
                network = Network(data_loader=self.dataLoader, is_autoencoder=is_autoencoder,
                                  size_hidden_layer_1=size_hidden_layer_1, size_hidden_layer_2=size_hidden_layer_2)
                num_iterations = self.train_network(network=network, training_set=training_set, tuning_set=tuning_set,
                                                    learning_rate=learning_rate, max_iterations=max_iterations,
                                                    check_increment=check_increment)

                # Get the network's performance
                network_loss = network.calc_loss(testing_set=testing_set)

                # Also calculate hit rate
                if self.dataset in self.dataLoader.categorizationSets:
                    hit_rate = network.calc_hit_rate(testing_set=testing_set)
                else:
                    hit_rate = None

                # Calculate null model
                null_model = network.calc_null_model(testing_set=testing_set)

                # Create table to write
                dt_to_write = pd.DataFrame({
                    'learningRate': [learning_rate],
                    'hiddenLayerSize1': [size_hidden_layer_1],
                    'hiddenLayerSize2': [size_hidden_layer_2],
                    'max_iterations': [max_iterations],
                    'check_increment': [check_increment],
                    'iterationsToTrain': [num_iterations],
                    'networkLoss': [network_loss],
                    'hit_rate': [hit_rate],
                    'null_model': [null_model]
                })

                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    print(dt_to_write)

                # Write new file/append to file
                file_name = "data/results/" + network_type + "/" + self.dataset + str(iteration_ind + 1) + ".csv"
                if inputInd == 0 or not os.path.isfile(file_name):
                    dt_to_write.to_csv(file_name, index=False, header=True)
                else:
                    dt_to_write.to_csv(file_name, mode='a', index=False, header=False)

    def tune_appended_autoencoder(self, autoencoder_learning_rate, autoencoder_hidden_layer_size,
                                  learning_rates, hidden_layer_sizes, iterations=5):
        """
        Performs _x2 cross validation for an autoencoder appended to a classification/regression network

        :param autoencoder_learning_rate: (numeric) The learning rate by which to train the autoencoder
        :param autoencoder_hidden_layer_size: (int) The number of nodes to put in the autoencoder's hidden layer
        :param learning_rates: (numeric) Vector of learning rates to test
        :param hidden_layer_sizes: (int) Vector of sizes for the appended hidden layer to test
        :param iterations: (int) Number of cross validations to perform
        :return: None
        """

        # For each iteration, split the training set and run each combination of parameters on both halves
        for iteration_ind in list(range(iterations * 2)):
            if iteration_ind % 2 == 0:
                train_set_num = 1
                test_set_num = 2
                # Only re-split the training data every odd run
                self.dataLoader.split_training_data()
            else:
                train_set_num = 2
                test_set_num = 1

            # Set datasets
            training_set = self.dataLoader.trainingData[self.dataLoader.trainingData["set"] == train_set_num].copy()
            testing_set = self.dataLoader.trainingData[self.dataLoader.trainingData["set"] == test_set_num].copy()
            tuning_set = self.dataLoader.tuningData.copy()

            # Normalize datasets
            training_set = self.dataLoader.normalize_data(data=training_set)
            testing_set = self.dataLoader.normalize_data(data=testing_set)
            tuning_set = self.dataLoader.normalize_data(data=tuning_set)

            # Set max iterations and check increments --> to ensure it doesn't run too long or too short
            max_iterations = len(training_set) * 200
            check_increment = len(training_set) * 2

            # Iterate through each input combination
            input_ind = 0
            for uniqueLearningRate in learning_rates:
                for uniqueHiddenSize in hidden_layer_sizes:

                    print("Dataset=" + self.dataset + "; Iteration=" + str(iteration_ind + 1) +
                          "; Learning Rate=" + str(uniqueLearningRate) + "; Hidden Layer 2 Size=" +
                          str(uniqueHiddenSize))

                    # Train an autoencoder
                    network = Network(data_loader=self.dataLoader, is_autoencoder=True,
                                      size_hidden_layer_1=autoencoder_hidden_layer_size, size_hidden_layer_2=None)
                    num_iterations = self.train_network(network=network, training_set=training_set,
                                                        tuning_set=tuning_set,
                                                        learning_rate=autoencoder_learning_rate,
                                                        max_iterations=max_iterations,
                                                        check_increment=check_increment)

                    # Append the autoencoder to a single hidden layer network
                    network.append_autoencoder_to_network(size_hidden_layer_2=uniqueHiddenSize)
                    num_iterations_add = self.train_network(network=network, training_set=training_set,
                                                            tuning_set=tuning_set,
                                                            learning_rate=uniqueLearningRate,
                                                            max_iterations=max_iterations,
                                                            check_increment=check_increment)
                    num_iterations = num_iterations + num_iterations_add

                    # Get the network's performance
                    network_loss = network.calc_loss(testing_set=testing_set)

                    # Also calculate hit rate
                    if self.dataset in self.dataLoader.categorizationSets:
                        hit_rate = network.calc_hit_rate(testing_set=testing_set)
                    else:
                        hit_rate = None

                    # Calculate null model
                    null_model = network.calc_null_model(testing_set=testing_set)

                    # Create table to write
                    dt_to_write = pd.DataFrame({
                        'autoencoderLearningRate': [autoencoder_learning_rate],
                        'autoencoderHiddenLayerSize': [autoencoder_hidden_layer_size],
                        'learningRate': [uniqueLearningRate],
                        'hiddenLayerSize2': [uniqueHiddenSize],
                        'max_iterations': [max_iterations],
                        'check_increment': [check_increment],
                        'iterationsToTrain': [num_iterations],
                        'networkLoss': [network_loss],
                        'hit_rate': [hit_rate],
                        'null_model': [null_model]
                    })

                    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                        print(dt_to_write)

                    # Write new file/append to file
                    file_name = "data/results/appended_autoencoder/" + self.dataset + str(iteration_ind + 1) + ".csv"
                    if input_ind == 0 or not os.path.isfile(file_name):
                        dt_to_write.to_csv(file_name, index=False, header=True)
                    else:
                        dt_to_write.to_csv(file_name, mode='a', index=False, header=False)

                    input_ind = input_ind + 1

    def summarize_results(self, network_type, print_pretty=False):
        """
        Gathers the results from the cross validations performed and aggregates them into a single table of results

        :param network_type: (String) 'autoencoder', 'autoencoder-appended', 'simple', or 'layered'
        :param print_pretty: (Boolean) Whether to print the summary table in Latex form for the paper
        :return: (DataFrame) The aggregated output table
        """
        # Base path of the files to read
        file_base = "data/results/" + network_type + "/" + self.dataset

        # Gathers all of the outputs
        full_dt = pd.DataFrame()

        # Iterate through each of the 10 cross validations and append them to the full table
        for ind in list(range(10)):
            file_name = file_base + str(ind + 1) + ".csv"
            if os.path.isfile(file_name):
                full_dt_add = pd.read_csv(file_name)
                full_dt_add['iteration'] = ind + 1

                full_dt = pd.concat([full_dt, full_dt_add], axis=0, ignore_index=True)

        # Aggregate the results by grouping and averaging
        group_cols = ['learningRate', 'hiddenLayerSize1', 'hiddenLayerSize2']
        cols_to_avg = ['max_iterations', 'check_increment', 'iterationsToTrain', 'networkLoss',
                       'hit_rate', 'null_model', 'autoencoderLearningRate', 'autoencoderHiddenLayerSize']

        full_dt = full_dt.dropna(axis=1, how='all')
        group_cols = [x for x in group_cols if x in full_dt.columns]
        cols_to_avg = [x for x in cols_to_avg if x in full_dt.columns]

        summary_dt = full_dt.groupby(group_cols)[cols_to_avg].mean().reset_index()

        # Print in Latex form
        if print_pretty:
            print_dt = summary_dt.copy()

            # Remove columns irrelevant to my paper
            print_dt = print_dt.drop(columns=['max_iterations', 'check_increment'])

            if type == 'autoencoder':
                drop_cols = ['hit_rate', 'null_model']
                drop_cols = [x for x in drop_cols if x in print_dt.columns]
                print_dt = print_dt.drop(columns=drop_cols)

            if type == 'appended_autoencoder':
                print_dt['autoencoderHiddenLayerSize'] = print_dt['autoencoderHiddenLayerSize'].astype(int)

            print_dt['iterationsToTrain'] = print_dt['iterationsToTrain'].astype(int)

            # Rename the columns
            col_dict = {'learningRate': 'Learn Rate', 'hiddenLayerSize1': 'N1', 'hiddenLayerSize2': 'N2',
                        'iterationsToTrain': 'Samples', 'networkLoss': 'Loss', 'hit_rate': 'Hit Rate',
                        'null_model': 'Null Model', 'autoencoderLearningRate': 'Autoencoder Rate',
                        'autoencoderHiddenLayerSize': 'Autoencoder N'}
            print_dt.rename(columns=col_dict, inplace=True)

            # Prints
            print(self.dataset)
            print(print_dt.to_latex(index=False))

        return summary_dt


if __name__ == '__main__':
    os.chdir('C:\\Users\\toddi\\PycharmProjects\\Programming-Project-3')

    # ------------------------- Video -------------------------
    # Train models
    # Categorization
    categorization_trainer = NetworkTuner(dataset="breast-cancer-wisconsin", reload_data=False)
    cat_data_loader = categorization_trainer.dataLoader
    cat_training_set = cat_data_loader.trainingData
    cat_tuning_set = cat_data_loader.tuningData
    cat_training_set = cat_data_loader.normalize_data(data=cat_training_set)
    cat_tuning_set = cat_data_loader.normalize_data(data=cat_tuning_set)
    categorization_network = Network(data_loader=cat_data_loader, is_autoencoder=False,
                                     size_hidden_layer_1=None, size_hidden_layer_2=None)
    categorization_trainer.train_network(network=categorization_network, training_set=cat_training_set,
                                         tuning_set=cat_tuning_set, learning_rate=.1,
                                         max_iterations=len(cat_training_set),
                                         check_increment=len(cat_training_set))
    # Regression
    regression_trainer = NetworkTuner(dataset="machine", reload_data=False)
    reg_data_loader = regression_trainer.dataLoader
    reg_training_set = reg_data_loader.trainingData
    reg_tuning_set = reg_data_loader.tuningData
    reg_training_set = reg_data_loader.normalize_data(data=reg_training_set)
    reg_tuning_set = reg_data_loader.normalize_data(data=reg_tuning_set)
    regression_network = Network(data_loader=reg_data_loader, is_autoencoder=False,
                                 size_hidden_layer_1=None, size_hidden_layer_2=None)
    regression_trainer.train_network(network=regression_network, training_set=reg_training_set,
                                     tuning_set=reg_tuning_set, learning_rate=.001,
                                     max_iterations=len(reg_training_set),
                                     check_increment=len(reg_training_set))

    # Autoencoder
    autoencoder_trainer = NetworkTuner(dataset="machine", reload_data=False)
    aut_data_loader = autoencoder_trainer.dataLoader
    aut_training_set = aut_data_loader.trainingData
    aut_tuning_set = aut_data_loader.tuningData
    aut_training_set = aut_data_loader.normalize_data(data=aut_training_set)
    aut_tuning_set = aut_data_loader.normalize_data(data=aut_tuning_set)
    autoencoder_network = Network(data_loader=aut_data_loader, is_autoencoder=True,
                                  size_hidden_layer_1=3, size_hidden_layer_2=None)
    autoencoder_trainer.train_network(network=autoencoder_network, training_set=aut_training_set,
                                      tuning_set=aut_tuning_set, learning_rate=.001,
                                      max_iterations=len(aut_training_set),
                                      check_increment=len(aut_training_set))

    # Show folds ----
    cat_loss = categorization_network.calc_loss(testing_set=cat_tuning_set)
    reg_loss = regression_network.calc_loss(testing_set=reg_tuning_set)
    aut_loss = autoencoder_network.calc_loss(testing_set=aut_tuning_set)

    print("Categorization Cross Entropy Loss = " + str(cat_loss))
    print("Regression Mean Squared Error = " + str(reg_loss))
    print("Autoencoder Reconstruction Loss = " + str(aut_loss))



    # Simple network
    # valid_datasets = ['machine', 'breast-cancer-wisconsin', 'car', 'forestfires', 'house-votes-84', 'abalone']
    # for validDataset in valid_datasets:
    #     network_tuner = NetworkTuner(dataset=validDataset, reload_data=False)
    #     network_tuner.tune_for_a_dataset(network_type='simple', iterations=5,
    #                                      learning_rates=[.0001, .001, .01, .1], hidden_layer_sizes1=None,
    #                                      hidden_layer_sizes2=None)
    #     summ_dt = network_tuner.summarize_results(network_type='simple')
    #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #         print(validDataset)
    #         print(summ_dt)

    # Multilayer network
    # valid_datasets = ['machine', 'breast-cancer-wisconsin', 'car', 'forestfires', 'house-votes-84', 'abalone']
    # learning_rates = [[.001, .01], [.01, .1], [.01, .1], [.0001, .001], [.01, .1], [.001, .01]]
    # # valid_datasets = ['forestfires', 'abalone', 'house-votes-84']
    # # learning_rates = [[.001], [.001], [.1]]
    # for validDatasetInd in list(range(len(valid_datasets))):
    #     validDataset = valid_datasets[validDatasetInd]
    #     learning_rates_cur = learning_rates[validDatasetInd]
    #     # learning_rates_cur = [.0001, .001, .01, .1]
    #     sampleDataLoader = DataLoader(dataset=validDataset)
    #     sampleDataLoader.load_saved_data()
    #     num_features = len(sampleDataLoader.trainingData.columns) - 2
    #     hidden_layer_sizes = [int(num_features / 4), int(num_features / 2), num_features,
    #                           num_features * 2]
    #     network_tuner = NetworkTuner(dataset=validDataset, reload_data=False)
    #     network_tuner.tune_for_a_dataset(network_type='multilayer', iterations=5,
    #                                      learning_rates=learning_rates_cur,
    #                                      hidden_layer_sizes1=hidden_layer_sizes,
    #                                      hidden_layer_sizes2=hidden_layer_sizes)
    #
    # for validDatasetInd in list(range(len(valid_datasets))):
    #     validDataset = valid_datasets[validDatasetInd]
    #     network_tuner = NetworkTuner(dataset=validDataset, reload_data=False)
    #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #         print(validDataset)
    #         print(network_tuner.summarize_results(network_type='multilayer'))


    # network_tuner = NetworkTuner(dataset='breast-cancer-wisconsin')
    # train_dt = network_tuner.dataLoader.normalize_data(data=network_tuner.dataLoader.trainingData)
    # tune_dt = network_tuner.dataLoader.normalize_data(data=network_tuner.dataLoader.tuningData.copy())
    #
    # new_network = Network(data_loader=network_tuner.dataLoader, is_autoencoder=False,
    #                       size_hidden_layer_1=5, size_hidden_layer_2=9)
    # initialHitRate = new_network.calc_hit_rate(testing_set=tune_dt)
    #
    # network_tuner.train_network(network=new_network, training_set=train_dt, tuning_set=tune_dt,
    #                             learning_rate=.1, max_iterations=1000000,
    #                             check_increment=len(train_dt) * 3)
    #
    # newHitRate = new_network.calc_hit_rate(testing_set=tune_dt)
    # print(initialHitRate)
    # print(newHitRate)

    # Autoencoders
    # valid_datasets = ['machine', 'breast-cancer-wisconsin', 'car', 'forestfires', 'house-votes-84', 'abalone']
    # learning_rates = [[.001, .01], [.01, .1], [.01, .1], [.0001, .001], [.01, .1], [.001, .01]]
    # for validDatasetInd in list(range(len(valid_datasets))):
    #     validDataset = valid_datasets[validDatasetInd]
    #     # learning_rates_cur = [.0001, .001, .01, .1]
    #     learning_rates_cur = learning_rates[validDatasetInd]
    #     sampleDataLoader = DataLoader(dataset=validDataset)
    #     sampleDataLoader.load_saved_data()
    #     num_features = len(sampleDataLoader.trainingData.columns) - 2
    #     hidden_layer_sizes = [int(num_features / 4), int(num_features / 2), int(num_features * .75)]
    #     network_tuner = NetworkTuner(dataset=validDataset, reload_data=False)
    #     network_tuner.tune_for_a_dataset(network_type='autoencoder', iterations=5,
    #                                      learning_rates=learning_rates_cur,
    #                                      hidden_layer_sizes1=hidden_layer_sizes,
    #                                      hidden_layer_sizes2=None)
    #
    # for validDatasetInd in list(range(len(valid_datasets))):
    #     validDataset = valid_datasets[validDatasetInd]
    #     network_tuner = NetworkTuner(dataset=validDataset, reload_data=False)
    #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #         print(validDataset)
    #         print(network_tuner.summarize_results(network_type='autoencoder'))

    # Appended Autoencoders
    # valid_datasets = ['machine', 'breast-cancer-wisconsin', 'car', 'forestfires', 'house-votes-84', 'abalone']
    # autoencoder_lrs = [.001, .1, .1, .001, .1, .01]
    # autoencoder_sizes = [4, 4, 11, 6, 8, 4]
    # learning_rates = [[.001], [.1], [.1], [.0001, .001], [.1], [.001, .01]]
    # hidden_layer_sizes_full = [[12, 24], [4, 9], [15, 30], [6, 13], [8, 16, 32], [2, 9]]
    # for validDatasetInd in list(range(len(valid_datasets))):
    #     validDataset = valid_datasets[validDatasetInd]
    #     learning_rates_cur = learning_rates[validDatasetInd]
    #     autoencoder_lr = autoencoder_lrs[validDatasetInd]
    #     autoencoder_size = autoencoder_sizes[validDatasetInd]
    #     sampleDataLoader = DataLoader(dataset=validDataset)
    #     sampleDataLoader.load_saved_data()
    #     num_features = len(sampleDataLoader.trainingData.columns) - 2
    #     hidden_layer_sizes = hidden_layer_sizes_full[validDatasetInd]
    #     # hidden_layer_sizes = [int(num_features / 4), int(num_features / 2), num_features,
    #     #                       num_features * 2, num_features * 4]
    #     network_tuner = NetworkTuner(dataset=validDataset, reload_data=False)
    #     network_tuner.tune_appended_autoencoder(autoencoder_learning_rate=autoencoder_lr,
    #                                             autoencoder_hidden_layer_size=autoencoder_size,
    #                                             learning_rates=learning_rates_cur,
    #                                             hidden_layer_sizes=hidden_layer_sizes, iterations=5)
    #
    # for validDatasetInd in list(range(len(valid_datasets))):
    #     validDataset = valid_datasets[validDatasetInd]
    #     network_tuner = NetworkTuner(dataset=validDataset, reload_data=False)
    #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #         network_tuner.summarize_results(network_type='appended_autoencoder', print_pretty=True)

