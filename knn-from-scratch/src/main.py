import os
from src.data.data_loader import DataLoader
from src.models.knn import NearestNeighbor2


if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

    # One-hot coding
    Loader1 = DataLoader('abalone')
    data = Loader1.load_data()
    print(data)
    data = Loader1.pre_process_data(data)
    data = Loader1.handle_nominal_data(data)
    print(data)
    Loader1.split_and_save_data(data)

    # Split into 20% tuning, 40% training 1, and 40% training 2
    Model1 = NearestNeighbor2('breast-cancer-wisconsin')
    print("Full number of rows: " + str(len(Model1.trainingData) + len(Model1.tuningData)))
    Model1.split_training_data()
    train_set1 = Model1.trainingData[Model1.trainingData['set'] == 1]
    train_set2 = Model1.trainingData[Model1.trainingData['set'] == 2]
    print("Tuning size: " + str(len(Model1.tuningData)))
    print(Model1.tuningData[Model1.predictor].value_counts())
    print("Training size 1: " + str(len(train_set1)))
    print(train_set1[Model1.predictor].value_counts())
    print("Training size 2: " + str(len(train_set2)))
    print(train_set2[Model1.predictor].value_counts())
    print()
    print()

    # Normalize
    print(train_set1.head(2))
    normalized_data = Model1.normalize_data(train_set1.copy())
    print(normalized_data.head(2))
    print()
    print()

    # Calculate distance
    dist1 = Model1.calc_distance(
        sample1=normalized_data.loc[[0]].copy(),
        sample2=normalized_data.loc[[1]].copy(),
        exponent=2
    )
    print("Distance 1 = " + str(dist1))
    dist2 = Model1.calc_distance(
        sample1=normalized_data.loc[[0]].copy(),
        sample2=normalized_data.loc[[0]].copy(),
        exponent=2
    )
    print("Distance 2 = " + str(dist2))
