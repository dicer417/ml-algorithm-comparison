import os.path
import pandas as pd
from src.models.knn import NearestNeighbor2
from src.data.data_loader import DataLoader


class Tuner3:
    """
    A class to tune models for each dataset
    """
    def __init__(self, dataset, k_vals, exponent_vals, standard_dev_mult_vals, reload_data=False,
                 file_location=None):
        """
        Initializes a Tuner object for running tuning tests on a specified dataset for the provided parameter values

        :param dataset: (String) Name of the dataset for which to tune a k-nearest-neighbors algorithm
        :param k_vals: (List of ints) The k values, i.e. the number of neighbors, to test
        :param exponent_vals: (List of ints) The exponent values, for the distance formulas, to test
        :param standard_dev_mult_vals: (List of ints) The scaling factors in the Gaussian Kernel function to test
        """

        # Set Object attributes
        self.dataset = dataset                                      # Name of the dataset
        self.loader = DataLoader(self.dataset,                      # DataLoader for this Tuner
                                 file_location=file_location)
        self.k_vals = k_vals                                        # k values to test for tuning
        self.exponent_vals = exponent_vals                          # exponent values to test for tuning
        self.standard_dev_mult_vals = standard_dev_mult_vals        # standard_dev_mult values to test for tuning
        self.categorizationSets = self.loader.categorizationSets    # Datasets meant for categorization, not regression
        self.estimates = pd.DataFrame()                             # Table of calculated estimates of the tuning set
        self.hit_rate_dt = pd.DataFrame()

        # Perform data loading, processing, splitting, and saving
        if reload_data is True:
            data = self.loader.load_data()
            data = self.loader.pre_process_data(data)
            data = self.loader.handle_nominal_data(data)
            self.loader.split_and_save_data(data)

    def get_tuning_predictions(self, iterations=5):
        """
        For each unique set of parameters, i.e. each unique set of [k, exponent, standard_dev_mult],
        runs 10 iterations, in line with 5x2 cross validation, of the k-nearest-neighbors classification or
        regression function.

        The predictions are saved in a csv and set as the "estimates" value of the Tuner.

        :param iterations: (int) The number of cross validation iterations to run,
        i.e. for [iterations]x2 cross validation
        :return: Null
        """

        run_num = 0  # For tracking each run when we save

        # Create a new model for the number of times requested
        for iteration_ind in list(range(0, iterations)):
            # Initialize nearest-neighbor model and split the training data
            model = NearestNeighbor2(self.dataset)
            model.split_training_data()
            model_data_raw = model.trainingData

            # Index for each training set in our [iterations]x2 cross validation
            for training_ind in [1, 2]:
                estimates_dt = pd.DataFrame()

                # Iterate run_num
                run_num = run_num + 1

                # Print for tracking
                print("Starting run #" + str(training_ind) + " for Model #" + str(iteration_ind + 1))

                # Get the training and tuning datasets
                model_data = model_data_raw[model_data_raw['set'] == training_ind].copy()
                test_data = model.tuningData.copy()

                # Normalize both datasets
                model_data = model.normalize_data(model_data)
                test_data = model.normalize_data(test_data)

                # Calculate values necessary for distance/Kernel functions
                if self.dataset in model.categorizationSets:
                    model.calc_nominal_frequencies(model_data)
                else:
                    model.calc_vol(model_data)

                for ind in list(range(0, len(test_data))):
                    for unique_exponent in self.exponent_vals:
                        # Simplify frequency table beforehand
                        if self.dataset in model.categorizationSets and len(model.freq_table) > 0:
                            model.simplify_freq_table(exponent=unique_exponent)

                        # Calculate distances to all samples in the training set
                        nearest_neighbors = model.calculate_neighbors(
                            test_data.iloc[[ind]].copy(),
                            model_data,
                            unique_exponent
                        )

                        for unique_k in self.k_vals:
                            for unique_standard_dev_mult in self.standard_dev_mult_vals:
                                # Print to show progress (so I don't go insane)
                                print("Run #" + str(training_ind) + " for Model #" + str(iteration_ind + 1) +
                                      ', Dataset=' + self.dataset + ', k=' + str(unique_k) +
                                      ', exponent=' + str(unique_exponent) + ', standard_dev_mult=' +
                                      str(unique_standard_dev_mult) + ', Sample #' + str(ind))

                                estimate_data_add = test_data.iloc[[ind]].copy()
                                estimate_data_add['tuning_run'] = run_num
                                estimate_data_add['k'] = unique_k
                                estimate_data_add['exponent'] = unique_exponent
                                estimate_data_add['standard_dev_mult'] = unique_standard_dev_mult

                                # For datasets marked for categorization, run classification.
                                # For others, run regression.
                                if self.dataset in model.categorizationSets:
                                    determined_category = model.determine_category(nearest_neighbors,
                                                                                   k=unique_k)
                                    estimate_data_add['estimate'] = determined_category
                                else:
                                    val_estimate = model.estimate_function_value(
                                        nearest_neighbors,
                                        k=unique_k,
                                        standard_dev_mult=unique_standard_dev_mult)
                                    estimate_data_add['estimate'] = val_estimate

                                # Add these estimates to the full running table
                                estimates_dt = pd.concat([estimates_dt, estimate_data_add], axis=0,
                                                         ignore_index=True)

                # Save estimates table as a csv
                directory_path = "data/estimates"
                if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
                    os.mkdir(directory_path)
                estimates_csv_name = (directory_path + '/' + self.dataset + '_run' + str(run_num) + '.csv')
                estimates_dt.to_csv(estimates_csv_name, index=False)

    def calculate_hit_rates(self, ests_type='tuning'):
        """
        Aggregates the predictions for a given dataset and a specific tuning/testing step and calculates the hit rate
        for each parameter set.

        :param ests_type: Dictates the location of the files to load
        :return:
        """
        # Get the names of the files that contain the estimates
        files = []
        sum_cols = ['hit', 'count']
        base_dir = 'data/'
        if ests_type == 'tuning':
            if self.dataset in self.categorizationSets:
                group_cols = ['k', 'exponent']
            else:
                group_cols = ['k', 'exponent', 'standard_dev_mult']
            base_dir = base_dir + 'estimates/'
            for ind in list(range(1, 11)):
                file_name = base_dir + self.dataset + "_run" + str(ind) + ".csv"
                files.append(file_name)
        elif ests_type == 'editing':
            if self.dataset in self.categorizationSets:
                group_cols = ['portion_to_leave']
            else:
                group_cols = ['portion_to_leave', 'error_threshold']
            base_dir = base_dir + 'estimates_edited/'
            for ind in list(range(1, 11)):
                file_name = base_dir + self.dataset + "_run" + str(ind) + ".csv"
                files.append(file_name)
        else:
            group_cols = []
            base_dir = base_dir + 'testing/'
            for ind in list(range(1, 11)):
                file_name = base_dir + self.dataset + "_run" + str(ind) + ".csv"
                files.append(file_name)
        estimates_dt_summed = pd.DataFrame()
        for unique_file in files:
            if os.path.exists(unique_file):
                estimates_dt = pd.read_csv(unique_file)
                estimates_dt['hit'] = (estimates_dt[self.loader.predictor] == estimates_dt['estimate'])
                estimates_dt['hit'] = estimates_dt['hit'].astype(int)
                estimates_dt['count'] = 1
                if len(group_cols) > 0:
                    estimates_dt_summed_add = estimates_dt.groupby(group_cols)[sum_cols].sum().reset_index()
                else:
                    estimates_dt_summed_add = pd.DataFrame(estimates_dt[sum_cols].sum()).transpose()

                # Add these estimates to the full running table
                estimates_dt_summed = pd.concat([estimates_dt_summed, estimates_dt_summed_add], axis=0,
                                                ignore_index=True)
        estimates_dt_summed['accuracy'] = estimates_dt_summed['hit'] / estimates_dt_summed['count']
        self.estimates = estimates_dt_summed

        # Average the accuracies of each parameter combination
        if len(group_cols) > 0:
            hit_rate_dt = estimates_dt_summed.groupby(group_cols)[['accuracy']].mean().reset_index()
        else:
            hit_rate_dt = pd.DataFrame(estimates_dt_summed[['accuracy']].mean()).transpose()

        # Add the Null Model Predictor
        model_null = NearestNeighbor2(self.dataset)
        model_null_data = model_null.trainingData
        null_val = model_null_data[model_null.predictor].mode()[0]
        null_accuracy = len(model_null_data[model_null_data[model_null.predictor] == null_val]) / len(model_null_data)
        hit_rate_dt['Null Model'] = null_accuracy

        # Print the LaTeX code
        print(hit_rate_dt.to_latex(index=False))
        print(hit_rate_dt)

        self.hit_rate_dt = hit_rate_dt

    def calculate_mean_squared_errors(self, ests_type='tuning'):
        """
        Aggregates the predictions for a given dataset and a specific tuning/testing step and calculates the MSE
        for each parameter set.

        :param ests_type: Dictates the location of the files to load
        :return:
        """
        # Get the names of the files that contain the estimates
        files = []
        base_dir = 'data/'
        sum_cols = ['squared_error', 'null_model', 'count']
        if ests_type == 'tuning':
            if self.dataset in self.categorizationSets:
                group_cols = ['k', 'exponent']
            else:
                group_cols = ['k', 'exponent', 'standard_dev_mult']
            base_dir = base_dir + 'estimates/'
            for ind in list(range(1, 11)):
                file_name = base_dir + self.dataset + "_run" + str(ind) + ".csv"
                files.append(file_name)
        elif ests_type == 'editing':
            if self.dataset in self.categorizationSets:
                group_cols = ['portion_to_leave']
            else:
                group_cols = ['portion_to_leave', 'error_threshold']
            base_dir = base_dir + 'estimates_edited/'
            for ind in list(range(1, 11)):
                file_name = base_dir + self.dataset + "_run" + str(ind) + ".csv"
                files.append(file_name)
        else:
            group_cols = []
            base_dir = base_dir + 'testing/'
            for ind in list(range(1, 11)):
                file_name = base_dir + self.dataset + "_run" + str(ind) + ".csv"
                files.append(file_name)

        # Get the Null Model value
        null_model = NearestNeighbor2(self.dataset)
        null_val = null_model.trainingData[null_model.predictor].mean()

        estimates_dt_summed = pd.DataFrame()
        for unique_file in files:
            if os.path.exists(unique_file):
                estimates_dt = pd.read_csv(unique_file)
                estimates_dt['squared_error'] = (
                        (estimates_dt[self.loader.predictor] - estimates_dt['estimate']) ** 2)
                estimates_dt['null_model'] = ((estimates_dt[self.loader.predictor] - null_val) ** 2)
                estimates_dt['count'] = 1
                if len(group_cols) > 0:
                    estimates_dt_summed_add = estimates_dt.groupby(group_cols)[sum_cols].sum().reset_index()
                else:
                    # estimates_dt_summed_add = estimates_dt[sum_cols].sum().reset_index()
                    estimates_dt_summed_add = pd.DataFrame(estimates_dt[sum_cols].sum()).transpose()

                # Add these estimates to the full running table
                estimates_dt_summed = pd.concat([estimates_dt_summed, estimates_dt_summed_add], axis=0,
                                                ignore_index=True)
        estimates_dt_summed['mean_squared_error'] = estimates_dt_summed['squared_error'] / estimates_dt_summed['count']
        estimates_dt_summed['Null Model'] = estimates_dt_summed['null_model'] / estimates_dt_summed['count']
        self.estimates = estimates_dt_summed

        # Average the accuracies of each parameter combination
        if len(group_cols) > 0:
            hit_rate_dt = estimates_dt_summed.groupby(
                group_cols)[['mean_squared_error', 'Null Model']].mean().reset_index()
        else:
            # hit_rate_dt = estimates_dt_summed[['mean_squared_error', 'Null Model']].mean().reset_index()
            hit_rate_dt = pd.DataFrame(estimates_dt_summed[['mean_squared_error', 'Null Model']].mean()).transpose()

        # Print the LaTeX code
        print(hit_rate_dt.to_latex(index=False))
        print(hit_rate_dt)

        self.hit_rate_dt = hit_rate_dt

    def test_model(self, k, exponent, standard_dev_mult, portion_to_leave=None, error_threshold=None, iterations=5):
        """
        Tests each half of the training set against the other, saving the results in data/testing

        :param k: (int) Number of neighbors, tuned
        :param exponent: (int) p, tuned
        :param standard_dev_mult: (int) lambda, tuned
        :param portion_to_leave: (int) d, tuned - the portion of the dataset to leave after editing
        :param error_threshold: (int) epsilon, tuned - the threshold for a regression value to be deemed correct
        :param iterations: (int) The number of tests to run
        :return: Null
        """
        # Create a new model for the number of times requested
        for iteration_ind in list(range(0, iterations)):
            estimates_dt = pd.DataFrame()

            # Initialize nearest-neighbor model and split the training data
            model = NearestNeighbor2(self.dataset)
            model.split_training_data()
            model_data_raw = model.trainingData

            model_data1 = model_data_raw[model_data_raw['set'] == 1].copy()
            model_data2 = model_data_raw[model_data_raw['set'] == 2].copy()

            # Normalize the training sets
            model_data1 = model.normalize_data(model_data1)
            model_data2 = model.normalize_data(model_data2)

            # Reduce the training set by the tuned amount
            if portion_to_leave is not None:
                if self.dataset in model.categorizationSets:
                    model.calc_nominal_frequencies(model_data1)
                else:
                    model.calc_vol(model_data1)

                # Simplify frequency table beforehand
                if self.dataset in model.categorizationSets and len(model.freq_table) > 0:
                    model.simplify_freq_table(exponent=exponent)
                model.trainingData = model_data1
                model.edit_training_data(portion_to_leave=portion_to_leave,
                                         error_threshold=error_threshold,
                                         k=1)  # Only use k=1 for the editing
                training_data1 = model.trainingData_edited.copy()
                model.trainingData = model_data2
                model.edit_training_data(portion_to_leave=portion_to_leave,
                                         error_threshold=error_threshold,
                                         k=1)  # Only use k=1 for the editing
                training_data2 = model.trainingData_edited.copy()

            else:
                training_data1 = model_data1
                training_data2 = model_data2

            # Calculate values necessary for distance/Kernel functions
            if self.dataset in model.categorizationSets:
                model.calc_nominal_frequencies(model_data1)
            else:
                model.calc_vol(model_data1)

            # Simplify frequency table beforehand
            if self.dataset in model.categorizationSets and len(model.freq_table) > 0:
                model.simplify_freq_table(exponent=exponent)

            # First, test the samples in set 2 on the training set 1
            estimates1 = []
            estimate_data_add = model_data2.copy()
            for ind in list(range(0, len(model_data2))):
                # Print to show progress (so I don't go insane)
                print("Model #" + str(iteration_ind) + ' for Set #1, Dataset=' + self.dataset +
                      ', Sample #' + str(ind))

                # Calculate distances to all samples in the training set
                nearest_neighbors = model.calculate_neighbors(
                    model_data2.iloc[[ind]].copy(),
                    training_data1,
                    exponent
                )

                # For datasets marked for categorization, run classification.
                # For others, run regression.
                if self.dataset in model.categorizationSets:
                    determined_category = model.determine_category(nearest_neighbors, k=k)
                    estimates1.append(determined_category)
                else:
                    val_estimate = model.estimate_function_value(
                        nearest_neighbors, k=k, standard_dev_mult=standard_dev_mult)
                    estimates1.append(val_estimate)

            # Add these estimates to the full running table
            estimate_data_add['estimate'] = estimates1
            estimate_data_add['run_num'] = iteration_ind
            estimates_dt = pd.concat([estimates_dt, estimate_data_add], axis=0,
                                     ignore_index=True)

            # Overwrite values necessary for distance/Kernel functions with those for set 2
            if self.dataset in model.categorizationSets:
                model.calc_nominal_frequencies(model_data2)
            else:
                model.calc_vol(model_data2)

            # Then, test the samples in set 1 on the training set 2
            estimates2 = []
            estimate_data_add = model_data1.copy()
            for ind in list(range(0, len(model_data1))):
                # Print to show progress (so I don't go insane)
                print("Model #" + str(iteration_ind) + ' for Set #2, Dataset=' + self.dataset +
                      ', Sample #' + str(ind))

                # Simplify frequency table beforehand
                if self.dataset in model.categorizationSets and len(model.freq_table) > 0:
                    model.simplify_freq_table(exponent=exponent)

                # Calculate distances to all samples in the training set
                nearest_neighbors = model.calculate_neighbors(
                    model_data1.iloc[[ind]].copy(),
                    training_data2,
                    exponent
                )

                # For datasets marked for categorization, run classification.
                # For others, run regression.
                if self.dataset in model.categorizationSets:
                    determined_category = model.determine_category(nearest_neighbors, k=k)
                    estimates2.append(determined_category)
                else:
                    val_estimate = model.estimate_function_value(
                        nearest_neighbors, k=k, standard_dev_mult=standard_dev_mult)
                    estimates2.append(val_estimate)

            # Add these estimates to the full running table
            estimate_data_add['estimate'] = estimates2
            estimate_data_add['run_num'] = iteration_ind
            estimates_dt = pd.concat([estimates_dt, estimate_data_add], axis=0,
                                     ignore_index=True)

            # Save the estimates
            estimates_file_save_loc = "data/testing/" + self.dataset + "_run" + str(iteration_ind + 1) + ".csv"
            estimates_dt.to_csv(estimates_file_save_loc, index=False)

    def tune_edited_knn(self, portions_to_leave, error_thresholds, k, exponent, std_dev_mult=1, iterations=3):
        """
        Classifies each sample in the tuning set 'iterations' times for each unique combination of
        portion_to_leave and error_threshold passed

        :param portions_to_leave: (int) The portion of the dataset to leave at the end of editing
        :param error_thresholds: (int) The cutoff for a regression prediction to be deemed correct
        :param k: (int) The number of neighbors to use for prediction
        :param exponent: (int) The degree of the Minskowki distance metric
        :param std_dev_mult: (int) The multiplier of the standard deviation in the Gaussian kernel function
        :param iterations: (int) Number of times to run for each parameter set
        :return:
        """
        # Create a new model for the number of times requested
        for iteration_ind in list(range(0, iterations)):
            # Initialize nearest-neighbor model and split the training data
            model = NearestNeighbor2(self.dataset)

            # Normalize the training data prior to pruning it
            model.trainingData = model.normalize_data(model.trainingData)

            # Calculate values necessary for distance/Kernel functions
            if self.dataset in model.categorizationSets:
                model.calc_nominal_frequencies(model.trainingData)
            else:
                model.calc_vol(model.trainingData)

            # Simplify frequency table beforehand
            if self.dataset in model.categorizationSets and len(model.freq_table) > 0:
                model.simplify_freq_table(exponent=exponent)

            estimates_dt = pd.DataFrame()

            # Edit model data downwards for each portion_to_leave passed
            for unique_portion_to_leave in portions_to_leave:
                for unique_error_threshold in error_thresholds:
                    # Print for tracking
                    print("Starting Model #" + str(iteration_ind + 1) + ", portion_to_leave: " +
                          str(unique_portion_to_leave) + ", error_threshold:" + str(unique_error_threshold))

                    # Edit the training set with the parameters to test
                    model.edit_training_data(portion_to_leave=unique_portion_to_leave,
                                             error_threshold=unique_error_threshold)

                    # Get the training and tuning datasets
                    model_data = model.trainingData_edited.copy()
                    test_data = model.tuningData.copy()

                    # Normalize test data datasets
                    test_data = model.normalize_data(test_data)

                    estimates = []
                    for ind in list(range(0, len(test_data))):
                        # Print to show progress (so I don't go insane)
                        print("Model #" + str(iteration_ind + 1) +
                              ', Dataset=' + self.dataset + ', portion_to_leave=' + str(unique_portion_to_leave) +
                              ', error_threshold=' + str(unique_error_threshold) + ', Sample #' + str(ind))

                        # Calculate distances to all samples in the training set
                        nearest_neighbors = model.calculate_neighbors(
                            test_data.iloc[[ind]].copy(),
                            model_data,
                            exponent=exponent
                        )

                        # For datasets marked for categorization, run classification.
                        # For others, run regression.
                        if self.dataset in model.categorizationSets:
                            # We edit the dataset with k=1 but use the tuned k here
                            determined_category = model.determine_category(nearest_neighbors, k=k)
                            estimates.append(determined_category)
                        else:
                            val_estimate = model.estimate_function_value(
                                nearest_neighbors,
                                k=k,  # We edit the dataset with k=1 but use the tuned k here
                                standard_dev_mult=std_dev_mult)
                            estimates.append(val_estimate)

                    # Create estimates table to append to the full table
                    estimate_data_add = test_data.copy()
                    estimate_data_add['estimate'] = estimates
                    estimate_data_add['portion_to_leave'] = unique_portion_to_leave
                    estimate_data_add['error_threshold'] = unique_error_threshold

                    # Add these estimates to the full running table
                    estimates_dt = pd.concat([estimates_dt, estimate_data_add], axis=0,
                                             ignore_index=True)

            # Save estimates table as a csv
            directory_path = "data/estimates_edited"
            if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
                os.mkdir(directory_path)
            estimates_csv_name = (directory_path + '/' + self.dataset + '_run' + str(iteration_ind + 1) + '.csv')
            estimates_dt.to_csv(estimates_csv_name, index=False)


if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

    # Testing k up to SQRT of the size of each training set, dividing by 4
    # --------------------------------------------------------------------
    # tuner = Tuner3('breast-cancer-wisconsin', [4, 8, 12, 16], [1, 2], [1],
    #                reload_data=False)
    # # tuner.get_tuning_predictions(iterations=5)
    # tuner.calculate_hit_rates(ests_type='tuning')
    #
    # tuner2 = Tuner3('house-votes-84', [3, 6, 9, 12], [1, 2], [1],
    #                 reload_data=False)
    # # tuner2.get_tuning_predictions(iterations=5)
    # tuner2.calculate_hit_rates(ests_type='tuning')
    # print(tuner2.hit_rate_dt)
    #
    # tuner3 = Tuner3('car', k_vals=[6, 12, 18, 24], exponent_vals=[1, 2],
    #                 standard_dev_mult_vals=[1], reload_data=False)
    # # tuner3.get_tuning_predictions(iterations=5)
    # tuner3.calculate_hit_rates(ests_type='tuning')
    # print(tuner3.hit_rate_dt)
    #
    # tuner4 = Tuner3('abalone', [10, 20, 30, 40], [1, 2], [1, 2],
    #                 reload_data=False)
    # # tuner4.get_tuning_predictions(iterations=5)  # Takes 4 hours
    # tuner4.calculate_mean_squared_errors(ests_type='tuning')
    # print(tuner4.hit_rate_dt)
    #
    # tuner5 = Tuner3('forestfires', [3, 6, 9, 12], [1, 2], [1, 2],
    #                 reload_data=False)
    # # tuner5.get_tuning_predictions(iterations=5)
    # tuner5.calculate_mean_squared_errors(ests_type='tuning')
    # print(tuner5.hit_rate_dt)
    #
    # tuner6 = Tuner3('machine', [2, 4, 6, 8], [1, 2], [1, 2],
    #                 reload_data=False)
    # # tuner6.get_tuning_predictions(iterations=5)
    # tuner6.calculate_mean_squared_errors(ests_type='tuning')
    # print(tuner6.hit_rate_dt)

    # Tune edited nearest neighbors
    # --------------------------------------------------------------
    # tuner1 = Tuner3('breast-cancer-wisconsin', [4, 8, 12, 16], [1, 2], [1],
    #                reload_data=False)
    # # tuner1.tune_edited_knn(k=12, exponent=2, portions_to_leave=[.25, .5, .75],
    # #                        error_thresholds=[.0001], iterations=3)
    # tuner1.calculate_hit_rates(ests_type='editing')

    # tuner2 = Tuner3('house-votes-84', [4, 8, 12, 16], [1, 2], [1],
    #                 reload_data=False)
    # # tuner2.tune_edited_knn(k=9, exponent=2, portions_to_leave=[.25, .5, .75],
    # #                        error_thresholds=[.0001], iterations=3)
    # tuner2.calculate_hit_rates(ests_type='editing')

    # tuner5 = Tuner3('forestfires', [4, 8, 12, 16], [1, 2], [1],
    #                 reload_data=False)
    # # tuner5.tune_edited_knn(k=9, exponent=2, std_dev_mult=2, portions_to_leave=[.25, .5, .75],
    # #                        error_thresholds=[.04, .07, .1], iterations=1)
    # tuner5.calculate_mean_squared_errors(ests_type='editing')

    # tuner6 = Tuner3('machine', [4, 8, 12, 16], [1, 2], [1],
    #                 reload_data=False)
    # # tuner6.tune_edited_knn(k=2, exponent=2, std_dev_mult=2, portions_to_leave=[.25, .5, .75],
    # #                        error_thresholds=[5, 10, 15], iterations=3)
    # tuner6.calculate_mean_squared_errors(ests_type='editing')

    # tuner3 = Tuner3('car', [4, 8, 12, 16], [1, 2], [1],
    #                reload_data=False)
    # tuner3.tune_edited_knn(k=6, exponent=2, portions_to_leave=[.25, .5, .75],
    #                        error_thresholds=[.0001], iterations=3)
    # tuner3.calculate_hit_rates(ests_type='editing')
    #
    # tuner4 = Tuner3('abalone', [4, 8, 12, 16], [1, 2], [1],
    #                reload_data=False)
    # # # tuner4.tune_edited_knn(k=20, exponent=2, std_dev_mult=2,
    # # #                        portions_to_leave=[.25, .5, .75], error_thresholds=[.5, .7], iterations=3)
    # tuner4.calculate_mean_squared_errors(ests_type='editing')
    
    # Test each model using the optimum parameters determined
    # -------------------------------------------------------
    # tuner = Tuner3('breast-cancer-wisconsin', [4, 8, 12, 16], [1, 2], [1],
    #                reload_data=False)
    # # tuner.test_model(k=12, exponent=2, standard_dev_mult=2, portion_to_leave=.5, error_threshold=1, iterations=1)
    # tuner.calculate_hit_rates('testing')
    #
    # tuner2 = Tuner3('house-votes-84', [4, 8, 12, 16], [1, 2], [1],
    #                reload_data=False)
    # # tuner2.test_model(k=9, exponent=2, standard_dev_mult=2, portion_to_leave=.5, error_threshold=1, iterations=1)
    # tuner2.calculate_hit_rates('testing')
    #
    # tuner3 = Tuner3('car', [4, 8, 12, 16], [1, 2], [1],
    #                reload_data=False)
    # # tuner3.test_model(k=6, exponent=2, standard_dev_mult=2, portion_to_leave=.75, error_threshold=1, iterations=1)
    # tuner3.calculate_hit_rates('testing')
    #
    # tuner4 = Tuner3('machine', [4, 8, 12, 16], [1, 2], [1],
    #                 reload_data=False)
    # # tuner4.test_model(k=2, exponent=2, standard_dev_mult=2, portion_to_leave=.25, error_threshold=5, iterations=1)
    # tuner4.calculate_mean_squared_errors('testing')
    #
    # tuner5 = Tuner3('forestfires', [4, 8, 12, 16], [1, 2], [1],
    #                 reload_data=False)
    # # tuner5.test_model(k=12, exponent=2, standard_dev_mult=2, portion_to_leave=1, error_threshold=1, iterations=1)
    # tuner5.calculate_mean_squared_errors('testing')
    #
    # tuner6 = Tuner3('abalone', [4, 8, 12, 16], [1, 2], [1],
    #                 reload_data=False)
    # # tuner6.test_model(k=20, exponent=2, standard_dev_mult=2, portion_to_leave=.75, error_threshold=.5, iterations=1)
    # tuner6.calculate_mean_squared_errors('testing')
