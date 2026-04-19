import pandas as pd
import numpy as np
import os


class DataLoader:
    """
    A class to run the data pipeline for the 6 datasets for k-nearest-neighbors classification and regression
    """
    def __init__(self, dataset, file_location=None):
        """
        Initializes a DataLoader object for the given dataset, setting object attributes unique to the dataset

        :param dataset: (String) Name of the dataset
        :param file_location: (String) Location of the raw .data file
        """

        # Validate dataset
        valid_datasets = ['abalone', 'breast-cancer-wisconsin', 'car', 'forestfires', 'house-votes-84', 'machine']
        if dataset not in valid_datasets:
            raise ValueError(dataset + " is not a valid dataset name.")

        self.dataset = dataset
        self.categorizationSets = ['breast-cancer-wisconsin', 'car', 'house-votes-84']
        self.comesWithHeaders = False
        self.ordinalAttributes = []
        self.nominalAttributes = []
        self.numericAttributes = []

        # Set location to find flat file with raw data
        if file_location is None:
            self.filepath = "data/raw/" + dataset + ".data"
        else:
            self.filepath = file_location

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

    def load_data(self):
        """
        Reads the csv containing the dataset

        :return: (DataFrame) The dataset
        """
        if self.comesWithHeaders:
            header = 1
        else:
            header = None

        data_loaded = pd.read_csv(self.filepath, header=header)
        return data_loaded

    def assign_headers(self, data_to_assign):
        """
        Assigns headers to the dataset

        :param data_to_assign: (DataFrame) The dataset
        :return: (DataFrame) The dataset, now with headers
        """
        if self.dataset == 'abalone':
            headers = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
                       'Shucked weight', 'Viscera weight', 'Shell weight',
                       'Rings']
        elif self.dataset == 'breast-cancer-wisconsin':
            headers = ['Sample code number', 'Clump Thickness',
                       'Uniformity of Cell Size',
                       'Uniformity of Cell Shape',
                       'Marginal Adhesion',
                       'Single Epithelial Cell Size', 'Bare Nuclei',
                       'Bland Chromatin', 'Normal Nucleoli',
                       'Mitoses', 'Class']
        elif self.dataset == 'car':
            headers = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'CAR']
        elif self.dataset == 'forestfires':
            headers = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']
        elif self.dataset == 'house-votes-84':
            headers = ['Class Name', 'handicapped-infants',
                       'water-project-cost-sharing',
                       'adoption-of-the-budget-resolution',
                       'physician-fee-freeze', 'el-salvador-aid',
                       'religious-groups-in-schools',
                       'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
                       'mx-missile', 'immigration', 'synfuels-corporation-cutback',
                       'education-spending', 'superfund-right-to-sue',
                       'crime', 'duty-free-exports',
                       'export-administration-act-south-africa']
        else:
            headers = ['vendor name', 'Model name', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']

        data_to_assign.columns = headers

    def pre_process_data(self, data_to_process):
        """
        Processes the raw dataset for use in a k-nearest-neighbors algorithm

        :param data_to_process: (DataFrame) The dataset to process
        :return: (DataFrame) The dataset, now processed
        """

        # Assign headers
        self.assign_headers(data_to_process)
        # Fill missing values
        self.handle_missing_values(data_to_process)
        # Assign unique identifiers to each sample in the dataset
        self.assign_sample_num(data_to_process)
        # Performs dataset-specific processing
        data_to_process = self.dataset_specific_processing(data_to_process)

        return data_to_process

    def assign_sample_num(self, data_to_assign):
        """
        Assigns unique identifiers to each sample in the dataset

        :param data_to_assign: (DataFrame) The dataset
        :return: (DataFrame) The dataset, now with a 'sampleNum' unique identifier column
        """
        # To be used as our identifier for each sample
        data_to_assign['sampleNum'] = data_to_assign.reset_index().index

    def dataset_specific_processing(self, data):
        """
        Performs dataset-specific processing

        :param data: (DataFrame) The dataset
        :return: (DataFrame) The dataset, now processed by its specific rules
        """

        # Handling specific to the forest fires dataset
        if self.dataset == 'forestfires':
            # Apply log function to area to correct for clustering near 0
            data['area'] = np.log1p(data['area'])

            # Replace month and day with numeric values
            month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
                         'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                         'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
            day_map = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4,
                       'fri': 5, 'sat': 6, 'sun': 7}
            data['month'] = data['month'].map(month_map)
            data['day'] = data['day'].map(day_map)

        # Handling specific to the machine dataset
        elif self.dataset == 'machine':
            # Save the ERP values in a csv file
            written_cols = ['sampleNum', 'ERP']
            data[written_cols].to_csv('data/processed/erp.csv', index=False)

            # Remove columns that won't be used in the nearest neighbors calculation
            cols_to_remove = ['vendor name', 'Model name', 'ERP']
            data = data.drop(columns=cols_to_remove)

        # Handling specific to the breast cancer dataset
        elif self.dataset == 'breast-cancer-wisconsin':
            # Remove old unique identifier column
            data = data.drop(columns='Sample code number')

        return data

    def handle_missing_values(self, data):
        """
        Removes missing values from a dataset

        :param data: (DataFrame) The dataset
        :return: Null
        """

        # Do not remove missing values for the Congressional votes dataset since ? does not signify a missing value
        if self.dataset == 'house-votes-84':
            return

        # Iterate through each column in the dataset
        for column in data.columns:
            # Identify rows with missing values
            nan_indices = data[data[column].isna()].index
            question_mark_indices = data[data[column] == '?'].index
            indices_to_replace = nan_indices.union(question_mark_indices)

            if len(indices_to_replace) > 0:
                # Set missing numeric attributes equal to the mean of the column
                if column in self.numericAttributes:
                    column_mean = data[column].replace('?', np.nan).astype(float).mean()  # Convert '?' to NaN
                    data.loc[indices_to_replace, column] = column_mean

                # Set missing categorical attributes equal to the mode of the column
                elif column in self.nominalAttributes or column in self.ordinalAttributes:
                    column_mode = data[column].replace('?', np.nan).mode().iloc[0]  # Convert '?' to NaN
                    data.loc[indices_to_replace, column] = column_mode

                # For all other columns not addressed, drop rows with missing values
                else:
                    data = data[data[column] != '?']
                    data = data.dropna(subset=[column])  # Remove rows with NaN values

    def handle_nominal_data(self, data):
        """
        For non-categorization datasets, implements one hot coding for nominal variables

        :param data: (DataFrame) The dataset
        :return:  (DataFrame) The dataset, with Boolean columns replacing the nominal columns
        """
        if ((self.dataset not in self.categorizationSets) and
                (len(self.nominalAttributes) > 0)):
            data = pd.get_dummies(data=data,
                                  prefix=self.nominalAttributes,
                                  dummy_na=False,
                                  columns=self.nominalAttributes,
                                  drop_first=True)
        return data

    def split_and_save_data(self, data):
        """
        Splits the dataset into a training set and a tuning set and saves them to csv files

        :param data: (DataFrame) The dataset
        :return: Null
        """

        # Stratify by class for categorization sets
        if self.dataset in self.categorizationSets:
            tuning_set = pd.DataFrame()
            training_set = pd.DataFrame()
            for uniqueCategory in data[self.predictor].unique():
                cat_data = data[data[self.predictor] == uniqueCategory]
                # Shuffle the order of the category's samples
                cat_data = cat_data.sample(frac=1)

                # Split off 20% of the category's samples and add it to the tuning set
                tuning_ind = int(.2 * len(cat_data))
                tuning_set = pd.concat([tuning_set, cat_data[0:tuning_ind]],
                                       axis=0, ignore_index=True)

                # Add the remaining 80% to the training set
                training_set = pd.concat([training_set, cat_data[tuning_ind:len(cat_data)]],
                                         axis=0, ignore_index=True)

        # For non-categorization sets, simply split randomly
        else:
            data = data.sample(frac=1)  # Shuffle data

            tuning_ind = int(.2 * len(data))
            tuning_set = data[0:tuning_ind]
            training_set = data[tuning_ind:len(data)]

        # Write the tuning and training sets to csv files
        tuning_set_name = 'data/processed/' + self.dataset + '.tuning.csv'
        training_set_name = 'data/processed/' + self.dataset + '.training.csv'
        tuning_set.to_csv(tuning_set_name, index=False)
        training_set.to_csv(training_set_name, index=False)

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

if __name__ == '__main__':
    os.chdir('C:\\Users\\toddi\\PycharmProjects\\Programming-Project-1')
