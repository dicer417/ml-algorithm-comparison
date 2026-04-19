import pandas as pd
import math as math
import os


class NearestNeighbor2:
    """
    A class to run k-nearest-neighbors classification or regression
    """
    def __init__(self, dataset):
        """
        Initializes a NearestNeighbor object for the given dataset, setting object attributes unique to the dataset,
        including reading the training and tuning prepared by a DataLoader object

        :param dataset: (String) Name of the dataset
        """

        self.dataset = dataset
        self.categorizationSets = ['breast-cancer-wisconsin', 'car', 'house-votes-84']
        tuning_set_name = 'data/processed/' + self.dataset + '.tuning.csv'
        training_set_name = 'data/processed/' + self.dataset + '.training.csv'
        self.tuningData = pd.read_csv(tuning_set_name)
        self.trainingData = pd.read_csv(training_set_name)
        self.ordinalAttributes = []
        self.nominalAttributes = []
        self.numericAttributes = []
        self.freq_table = pd.DataFrame()
        self.simple_freq_table = pd.DataFrame()
        self.standard_dev = 0.0
        self.trainingData_edited = pd.DataFrame()

        if dataset == 'abalone':
            self.predictor = 'Rings'
            self.nominalAttributes = ['Sex']
            self.numericAttributes = [
                'Length', 'Diameter', 'Height', 'Whole weight',
                'Shucked weight', 'Viscera weight', 'Shell weight',
                'Rings'
            ]
        elif dataset == 'breast-cancer-wisconsin':
            self.predictor = 'Class'
            self.numericAttributes = [
                'Clump Thickness',
                'Uniformity of Cell Size',
                'Uniformity of Cell Shape',
                'Marginal Adhesion',
                'Single Epithelial Cell Size', 'Bare Nuclei',
                'Bland Chromatin', 'Normal Nucleoli',
                'Mitoses']
        elif dataset == 'car':
            self.predictor = 'CAR'
            self.nominalAttributes = ['buying', 'maint', 'doors',
                                      'persons', 'lug_boot', 'safety']
        elif dataset == 'forestfires':
            self.comesWithHeaders = True
            self.ordinalAttributes = ['month', 'day']
            self.numericAttributes = [
                'X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI',
                'temp', 'RH', 'wind', 'rain'
            ]
            self.predictor = 'area'
        elif dataset == 'house-votes-84':
            self.nominalAttributes = [
                'handicapped-infants',
                'water-project-cost-sharing',
                'adoption-of-the-budget-resolution',
                'physician-fee-freeze',
                'el-salvador-aid',
                'religious-groups-in-schools',
                'anti-satellite-test-ban',
                'aid-to-nicaraguan-contras',
                'mx-missile', 'immigration',
                'synfuels-corporation-cutback',
                'education-spending',
                'superfund-right-to-sue',
                'crime', 'duty-free-exports',
                'export-administration-act-south-africa'
            ]
            self.numericAttributes = []
            self.predictor = 'Class Name'
        else:
            self.predictor = 'PRP'
            self.numericAttributes = [
                'MYCT', 'MMIN', 'MMAX',
                'CACH', 'CHMIN', 'CHMAX'
            ]

    def split_training_data(self):
        """
        Splits the training dataset into 2 halves, marking them with a new column called 'set'

        :return: Nulls
        """

        data = self.trainingData

        if self.dataset in self.categorizationSets:
            set1 = pd.DataFrame()
            set2 = pd.DataFrame()
            for uniqueCategory in data[self.predictor].unique():
                cat_data = data[data[self.predictor] == uniqueCategory].copy()
                # Shuffle the order of the category's samples
                cat_data = cat_data.sample(frac=1)

                # Split off 50% of the category's samples and add it to the first set
                cutoff_ind = int(.5 * len(cat_data))
                set1 = pd.concat([set1, cat_data[0:cutoff_ind]],
                                 axis=0, ignore_index=True)

                # Add the remaining 50% to the second set
                set2 = pd.concat([set2, cat_data[cutoff_ind:len(cat_data)]],
                                 axis=0, ignore_index=True)
        else:
            data = data.sample(frac=1)

            cutoff_ind = int(.5 * len(data))
            set1 = data[0:cutoff_ind].copy()
            set2 = data[cutoff_ind:len(data)].copy()
        set1['set'] = 1
        set2['set'] = 2
        self.trainingData = pd.concat([set1, set2],
                                      axis=0, ignore_index=True)

    def normalize_data(self, data):
        """
        Normalizes numeric factors using z-score standardization

        :param data: (DataFrame) Training or tuning dataset
        :return: (DataFrame) The same training/tuning dataset but with its numeric attributes normalized
        """

        for uniqueNumCol in self.numericAttributes:
            col_mean = data[uniqueNumCol].mean()
            col_sd = data[uniqueNumCol].std()

            data[uniqueNumCol] = (data[uniqueNumCol] - col_mean) / col_sd

        return data

    def calc_nominal_frequencies(self, data):
        """
        Calculates the nominal frequencies of categorical data for each class, in preparation of VDM calcs

        :param data: (DataFrame) Training dataset
        :return: Null
        """

        # Do not calculate nominal frequencies except for k-nearest neighbors sets
        if self.dataset not in self.categorizationSets:
            return

        freq_table = pd.DataFrame()
        for uniqueNominalCol in self.nominalAttributes:
            col_categories = data[uniqueNominalCol].unique()
            for uniqueCat in col_categories:
                cat_num = len(data[data[uniqueNominalCol] == uniqueCat])
                for uniqueClass in data[self.predictor].unique():
                    criteria = ((data[self.predictor] == uniqueClass) &
                                (data[uniqueNominalCol] == uniqueCat))
                    cat_class_num = len(data[criteria])
                    table_to_add = pd.DataFrame({
                        'nominalCol': [uniqueNominalCol],
                        'category': [uniqueCat],
                        'class': [uniqueClass],
                        'frequency': [cat_class_num / cat_num]})
                    freq_table = pd.concat([freq_table, table_to_add],
                                           axis=0, ignore_index=True)

        self.freq_table = freq_table

    def simplify_freq_table(self, exponent):
        freq_table = self.freq_table

        simple_freq_table = pd.DataFrame()

        for uniqueNominalCol in freq_table['nominalCol'].unique():
            unique_col_vals = freq_table['category'].unique()
            for uniqueColVal in unique_col_vals:
                for uniqueColVal2 in unique_col_vals:
                    # Get the frequency table calculated earlier to compute the Value Distance Metric
                    freq_dt1 = freq_table[freq_table['nominalCol'] == uniqueNominalCol].copy()
                    freq_dt1 = freq_dt1[freq_dt1['category'] == uniqueColVal]
                    freq_dt2 = freq_table[freq_table['nominalCol'] == uniqueNominalCol].copy()
                    freq_dt2 = freq_dt2[freq_dt2['category'] == uniqueColVal2]

                    # Merge the 2 tables to calculate the Value Distance Metric
                    freq_dt1.rename(columns={'frequency': 'freq1', 'category': 'category1'}, inplace=True)
                    freq_dt2.rename(columns={'frequency': 'freq2', 'category': 'category2'}, inplace=True)
                    comp_dt = pd.merge(freq_dt1, freq_dt2, on=['nominalCol', 'class'], how='outer')

                    # Remove NaN's
                    comp_dt.fillna({'freq1': 0.0, 'freq2': 0.0}, inplace=True)

                    # Calculate differences, exponent, and sum --> VDM
                    comp_dt['abs_diff'] = abs(comp_dt['freq1'] - comp_dt['freq2']) ** exponent

                    # Trim to necessary columns and append
                    comp_dt_summed = (
                        comp_dt.groupby(['nominalCol', 'category1', 'category2'])['abs_diff'].sum().reset_index())
                    simple_freq_table = pd.concat([simple_freq_table, comp_dt_summed], axis=0,
                                                  ignore_index=True)

        self.simple_freq_table = simple_freq_table

    def calc_vol(self, data):
        """
        Calculates the volatility of the attribute we are trying to predict and sets it as an Object attribute

        :param data: (DataFrame) Training set data
        :return: Null
        """
        self.standard_dev = data[self.predictor].std()

    def calc_distance(self, sample1, sample2, exponent):
        """
        Calculates the distance between 2 samples

        :param sample1: (DataFrame) A DataFrame of the first sample's factor values
        :param sample2: (DataFrame) A DataFrame of the second sample's factor values
        :param exponent: (int) The exponent to use in the Minkowski distance metric
        :return:
        """
        # Coerce exponent to float
        exponent = float(exponent)

        exclude_cols = ['set', 'sampleNum', self.predictor]  # Columns NOT to use for the distance calculation
        cols_for_distance = sample1.columns.tolist()         # Columns to use for the distance calculation
        cols_for_distance = [s for s in cols_for_distance if s not in exclude_cols]

        # Variables to track distances
        category_dist = 0.0
        num_dist = 0.0
        full_dist = 0.0

        # Iterate through each factor (i.e. variable i.e. column) and get its respective distance for the 2 samples
        for distanceCol in cols_for_distance:
            # Calculate distances for nominal attributes using VDM
            if distanceCol in self.nominalAttributes:
                freq_table = self.simple_freq_table[self.simple_freq_table['nominalCol'] == distanceCol]

                freq_table = freq_table[freq_table['category1'] == sample1[distanceCol].iloc[0]]
                freq_table = freq_table[freq_table['category2'] == sample2[distanceCol].iloc[0]]

                # Increment distance of categorical factors
                if len(freq_table) > 0:
                    category_dist = category_dist + freq_table['abs_diff'].iloc[0]
                # There's a chance that a classification for a factor (e.g. ? for a vote) in the tuning set does not
                # exist in the training set --> Use the maximum value
                else:
                    category_dist = category_dist + 1

            # Calculate distances for variables transformed with one hot coding
            elif pd.api.types.is_bool_dtype(sample1[distanceCol]):
                if bool(sample1[distanceCol].iloc[0]) != bool(sample2[distanceCol].iloc[0]):
                    num_dist = num_dist + 1

            # Calculate distances for numeric & ordinal variables
            else:
                num_dist_add = abs(sample1[distanceCol].iloc[0] - sample2[distanceCol].iloc[0]) ** exponent

                # Special treatment for month & day variables
                if distanceCol == 'month':
                    if 12 - num_dist_add < num_dist_add:
                        num_dist_add = 12 - num_dist_add
                    num_dist_add = num_dist_add / 6.0  # Min-max scaling
                elif distanceCol == 'day':
                    if 7 - num_dist_add < num_dist_add:
                        num_dist_add = 7 - num_dist_add
                    num_dist_add = num_dist_add / 3.5  # Min-max scaling

                num_dist_add = num_dist_add ** exponent  # Raise by exponent input
                num_dist = num_dist + num_dist_add  # Add to running total

            # Combine the categorical and numerical distances to get the full distance
            full_dist = (category_dist ** (1.0 / exponent)) + (num_dist ** (1.0 / exponent))

        return full_dist

    def calculate_neighbors(self, sample, data, exponent=1):
        """
        Finds the k-nearest-neighbors of a sample in a training dataset

        :param sample: (DataFrame) A DataFrame of the first sample's factor values
        :param data: (DataFrame) The training dataset
        :param exponent: (int) The exponent to use in the Minkowski distance metric
        :return:
        """

        # List to hold the distances of each training sample
        distances = []

        # Iterate through each training sample, calculate its distance from the tuning/testing sample, and append
        for rowInd in list(range(0, len(data))):
            sample2 = data.iloc[[rowInd]]
            distances.append(self.calc_distance(sample, sample2, exponent))

        # Set distances as a new column in the training dataset
        data['distance'] = distances

        # Sort by increasing distance
        sorted_data = data.sort_values(by='distance')

        # Use head() to get the top k results
        return sorted_data

    def determine_category(self, nearest_neighbors, k):
        """
        Returns the plurality of a sample's k-nearest-neighbors classification as its predicted value

        :param k: (int) The number of nearest neighbors to use
        :param nearest_neighbors: (DataFrame) A table of the k-nearest-neighbors of a sample, including their
        distances from the sample
        :return: (String) The predicted classification of a sample given its nearest neighbors
        """
        nearest_neighbors = nearest_neighbors.sort_values(by='distance')
        nearest_neighbors_prune = nearest_neighbors.head(k)

        return nearest_neighbors_prune[self.predictor].mode()[0]

    def estimate_function_value(self, nearest_neighbors, k, standard_dev_mult=2):
        """
        Estimates the numeric value of a sample given its k-nearest-neighbors

        :param k: (int) The number of nearest neighbors to use
        :param nearest_neighbors: (DataFrame) A table of the k-nearest-neighbors of a sample, including their
        distances from the sample
        :param standard_dev_mult: (int) The multiplier by which to scale the volatility scaler in the Gaussian Kernel
        function
        :return:
        """

        # Reduce nearest neighbors to the k nearest
        nearest_neighbors = nearest_neighbors.sort_values(by='distance')
        nearest_neighbors_prune = nearest_neighbors.head(k)

        # Variables to track the sums
        numerator = 0
        denominator = 0

        for ind in list(range(0, len(nearest_neighbors_prune))):
            # Calculate the Gaussian Kernel value
            distance = nearest_neighbors_prune['distance'].iloc[ind]
            value = nearest_neighbors_prune[self.predictor].iloc[ind]
            kernel_val = math.exp((1 / (standard_dev_mult * self.standard_dev)) * distance)

            # Increment the tracking variables
            numerator = numerator + (kernel_val * value)
            denominator = denominator + kernel_val

        return numerator / denominator

    def edit_training_data(self, portion_to_leave, error_threshold=None, k=1, exponent=2):
        """
        Edits the training set down to the indicated % of the original set or until it cannot prune the set more,
        saving it as a feature of the NearestNeighbor2 object

        :param portion_to_leave: (int) The target portion of the set to leave at the end of editing
        :param error_threshold: (int) Cutoff for a regression prediction to be deemed correct
        :param k: (int) Number of neighbors for classification/regression
        :param exponent: (int) Degree of the Minkowski distance
        :return:
        """
        training_dt = self.trainingData.copy()

        # Shuffle the table to prune
        # training_dt = training_dt.sample(frac=1).reset_index(drop=True)

        # Don't need to run the function if we're leaving all of the training data
        if portion_to_leave == 1:
            self.trainingData_edited = training_dt
            return

        num_to_leave = int(round(len(training_dt) * portion_to_leave, 0))

        row_ind = 0
        current_max = len(training_dt)
        inds_since_prune = 0
        while len(training_dt) > num_to_leave:
            print('Editing dataset at row' + str(row_ind))
            sample_to_check = training_dt.loc[[row_ind]].copy()
            sample_model_data = training_dt.drop(training_dt.index[row_ind]).reset_index(drop=True)
            sample_neighbors = self.calculate_neighbors(sample_to_check, sample_model_data, exponent=exponent)

            if self.dataset in self.categorizationSets:
                estimate = self.determine_category(nearest_neighbors=sample_neighbors, k=k)
                if estimate == sample_to_check[self.predictor].iloc[0]:
                    training_dt = training_dt.drop(training_dt.index[row_ind]).reset_index(drop=True)
                    inds_since_prune = 0
                else:
                    row_ind = row_ind + 1
                    inds_since_prune = inds_since_prune + 1
            else:
                estimate = self.estimate_function_value(nearest_neighbors=sample_neighbors, k=k, standard_dev_mult=1)
                if abs(estimate - sample_to_check[self.predictor].iloc[0]) < error_threshold:
                    training_dt = training_dt.drop(training_dt.index[row_ind]).reset_index(drop=True)
                    inds_since_prune = 0
                else:
                    row_ind = row_ind + 1
                    inds_since_prune = inds_since_prune + 1
            if row_ind >= len(training_dt):
                row_ind = 0
                if len(training_dt) == current_max:
                    print("CANNOT REDUCE THIS SET ANYMORE")
                    break
                current_max = len(training_dt)
            if inds_since_prune > len(training_dt):
                print("CANNOT REDUCE THIS SET ANYMORE")
                break

        self.trainingData_edited = training_dt


if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

    model = NearestNeighbor2('breast-cancer-wisconsin')
    model.split_training_data()
    model_data_raw = model.trainingData
    model_data = model_data_raw[model_data_raw['set'] == 1].copy()
    test_data = model.tuningData.copy()

    # Normalize both datasets
    model_data = model.normalize_data(model_data)
    test_data = model.normalize_data(test_data)

    # Pre-calc nominal frequencies for VDM
    model.calc_nominal_frequencies(model_data)

    nearest_neighbors = model.calculate_neighbors(
        test_data.iloc[[0]].copy(),
        model_data,
        2
    )
    pd.set_option('display.max_columns', None)
    print(test_data.iloc[[0]])
    print()
    print()
    print(nearest_neighbors.head(12))
    determined_category = model.determine_category(nearest_neighbors, k=12)
    print("CLASSIFICATION: " + str(determined_category))
    print()
    print()
    print()

    # REGRESSION
    model = NearestNeighbor2('machine')
    model.split_training_data()
    model_data_raw = model.trainingData
    model_data = model_data_raw[model_data_raw['set'] == 1].copy()
    test_data = model.tuningData.copy()

    # Normalize both datasets
    model_data = model.normalize_data(model_data)
    model.trainingData = model_data.copy()
    test_data = model.normalize_data(test_data)

    # Pre-calc predictor's standard dev
    model.calc_vol(model_data)

    nearest_neighbors = model.calculate_neighbors(
        test_data.iloc[[0]].copy(),
        model_data,
        2
    )
    print(test_data.iloc[[0]])
    print()
    print()
    print(nearest_neighbors.head(2))
    determined_category = model.estimate_function_value(nearest_neighbors, k=2, standard_dev_mult=2)
    print("REGRESSION: " + str(determined_category))
    print()
    print()
    print()

    # EDITING
    # model.trainingData = model.trainingData.sample(frac=1).reset_index(drop=True)
    print(model.trainingData.head(5))
    model.edit_training_data(portion_to_leave=.95, error_threshold=5, k=1, exponent=2)
    print(model.trainingData_edited.head(5))
