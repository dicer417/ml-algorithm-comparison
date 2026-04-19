import pandas as pd
import numpy as np
import os


class DataLoader:
    """
    A class to run the data pipeline for the 6 datasets for classification and regression
    """
    def __init__(self, dataset, file_location=None):
        """
        Initializes a DataLoader object for the given dataset, setting object attributes unique to the dataset

        :param dataset: (String) Name of the dataset
        :param file_location: (String) Location of the raw .data file. If None provided, defaults to data/raw/ path
        """

        # Validate dataset
        valid_datasets = ['abalone', 'breast-cancer-wisconsin', 'car', 'forestfires', 'house-votes-84', 'machine']
        if dataset not in valid_datasets:
            raise ValueError(dataset + " is not a valid dataset name.")

        # Set attributes
        self.dataset = dataset  # Name of the dataset
        self.categorizationSets = ['breast-cancer-wisconsin', 'car', 'house-votes-84']  # Classify, not regress
        self.comesWithHeaders = False           # For assigning headers
        self.ordinalAttributes = []             # Track attribute types
        self.nominalAttributes = []             # Track attribute types
        self.numericAttributes = []             # Track attribute types
        self.attributes = []                    # Track all attribute
        self.data = pd.DataFrame()              # For all the data
        self.trainingData = pd.DataFrame()      # Training dataset portion
        self.tuningData = pd.DataFrame()        # Tuning dataset portion

        # Set location to find flat file with raw data
        if file_location is None:
            self.filepath = "data/raw/" + dataset + ".data"
        else:
            self.filepath = file_location

        # Set unique attribute values for each dataset
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
            # self.ordinalAttributes = ['month', 'day']
            self.nominalAttributes = ['month', 'day']
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

        # Gather all attributes
        self.attributes = self.numericAttributes + self.ordinalAttributes + self.nominalAttributes

    def load_data(self):
        """
        Reads the csv containing the dataset and stores it in the object
        """
        if self.comesWithHeaders:
            header = 1
        else:
            header = None

        data_loaded = pd.read_csv(self.filepath, header=header)
        self.data = data_loaded

    def assign_headers(self):
        """
        Assigns headers to the dataset if necessary
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

        self.data.columns = headers

    def pre_process_data(self):
        """
        Processes the raw dataset for use in a classification or regression algorithm, except for normalization
        """

        # Assign headers
        self.assign_headers()
        # Fill missing values
        self.handle_missing_values()
        # One-hot coding for nominal data
        self.handle_nominal_data()
        # Assign unique identifiers to each sample in the dataset
        self.assign_sample_num()
        # Performs dataset-specific processing
        self.dataset_specific_processing()

    def assign_sample_num(self):
        """
        Assigns a unique sample number to each row in a dataset, acting as the primary key
        """
        # To be used as our identifier for each sample
        self.data['sampleNum'] = self.data.reset_index().index

    def dataset_specific_processing(self):
        """
        Perform processing specific to a given dataset, such as removing columns or applying a log function
        """
        # Handling specific to the forest fires dataset
        if self.dataset == 'forestfires':
            # Apply log function to area to correct for clustering near 0
            self.data['area'] = np.log1p(self.data['area'])

            # # Replace month and day with numeric values
            # month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
            #              'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
            #              'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
            # day_map = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4,
            #            'fri': 5, 'sat': 6, 'sun': 7}
            # self.data['month'] = self.data['month'].map(month_map)
            # self.data['day'] = self.data['day'].map(day_map)

        # Handling specific to the machine dataset
        elif self.dataset == 'machine':
            # Save the ERP values in a csv file
            written_cols = ['sampleNum', 'ERP']
            self.data[written_cols].to_csv('data/processed/erp.csv', index=False)

            # Remove columns that won't be used in the nearest neighbors calculation
            cols_to_remove = ['vendor name', 'Model name', 'ERP']
            self.data = self.data.drop(columns=cols_to_remove)

        # Handling specific to the breast cancer dataset
        elif self.dataset == 'breast-cancer-wisconsin':
            # Remove old unique identifier column
            self.data = self.data.drop(columns='Sample code number')
            self.data['Bare Nuclei'] = self.data['Bare Nuclei'].astype(int)

    def handle_missing_values(self):
        """
        Replace missing values with the mean or mode of the column
        """
        # Do not remove missing values for the Congressional votes dataset since ? does not signify a missing value
        if self.dataset == 'house-votes-84':
            return

        # Iterate through each column in the dataset
        for column in self.data.columns:
            # Identify rows with missing values
            nan_indices = self.data[self.data[column].isna()].index
            question_mark_indices = self.data[self.data[column] == '?'].index
            indices_to_replace = nan_indices.union(question_mark_indices)

            if len(indices_to_replace) > 0:
                # Set missing numeric attributes equal to the mean of the column
                if column in self.numericAttributes:
                    column_mean = self.data[column].replace('?', np.nan).astype(float).mean()  # Convert '?' to NaN
                    self.data.loc[indices_to_replace, column] = column_mean

                # Set missing categorical attributes equal to the mode of the column
                elif column in self.nominalAttributes or column in self.ordinalAttributes:
                    column_mode = self.data[column].replace('?', np.nan).mode().iloc[0]  # Convert '?' to NaN
                    self.data.loc[indices_to_replace, column] = column_mode

                # For all other columns not addressed, drop rows with missing values
                else:
                    data = self.data[self.data[column] != '?']
                    data = data.dropna(subset=[column])  # Remove rows with NaN values
                    self.data = data

    def handle_nominal_data(self):
        """
        For non-categorization datasets, implements one hot coding for nominal variables, updating the data attribute
        """
        data = self.data.copy()

        if len(self.nominalAttributes) > 0:
            data = pd.get_dummies(data=data,
                                  prefix=self.nominalAttributes,
                                  dummy_na=False,
                                  columns=self.nominalAttributes,
                                  drop_first=True)
        self.data = data

    def split_and_save_data(self):
        """
        Split the dataset into 80% training data and 20% tuning data, with stratification for classification sets,
        and save the resulting datasets
        """
        # Stratify by class for categorization sets
        if self.dataset in self.categorizationSets:
            tuning_set = pd.DataFrame()
            training_set = pd.DataFrame()
            for uniqueCategory in self.data[self.predictor].unique():
                cat_data = self.data[self.data[self.predictor] == uniqueCategory]
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
            data = self.data.sample(frac=1)  # Shuffle data

            tuning_ind = int(.2 * len(data))
            tuning_set = data[0:tuning_ind]
            training_set = data[tuning_ind:len(data)]

        # Write the tuning and training sets to csv files
        tuning_set_name = 'data/processed/' + self.dataset + '.tuning.csv'
        training_set_name = 'data/processed/' + self.dataset + '.training.csv'
        tuning_set.to_csv(tuning_set_name, index=False)
        training_set.to_csv(training_set_name, index=False)
        self.trainingData = training_set
        self.tuningData = tuning_set

    def load_saved_data(self):
        """
        Load training and tuning set that have already been processed and saved
        """
        tuning_set_name = 'data/processed/' + self.dataset + '.tuning.csv'
        training_set_name = 'data/processed/' + self.dataset + '.training.csv'
        self.tuningData = pd.read_csv(tuning_set_name)
        self.trainingData = pd.read_csv(training_set_name)

    def split_training_data(self):
        """
        Splits the training dataset into 2 halves, marking them with a new column called 'set'
        """

        data = self.trainingData

        # Drop any existing split
        if 'set' in data.columns:
            data = data.drop('set', axis=1)

        # Stratify by class
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


if __name__ == '__main__':
    os.chdir('C:\\Users\\toddi\\PycharmProjects\\Programming-Project-3')

    # datasets = ['abalone', 'breast-cancer-wisconsin', 'car', 'forestfires', 'house-votes-84', 'machine']
    # for unique_dataset in datasets:
    #     dl = DataLoader(unique_dataset)
    #     dl.load_data()
    #     dl.assign_headers()
    #     dl.pre_process_data()
    #     dl.split_and_save_data()
