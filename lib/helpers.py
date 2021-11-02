import pandas as pd
import numpy as np
import random
from typing import Tuple, Optional, List


def train_test_split(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    return train_df, test_df


def determine_type_of_feature(df: pd.DataFrame) -> List[str]:
    feature_types = []
    n_unique_values_treshold = 15
    for feature in df.columns:
        if feature != "label":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    return feature_types


def calculate_accuracy(predictions: pd.DataFrame, labels: pd.DataFrame) -> float:
    predictions_correct = predictions == labels
    accuracy = predictions_correct.mean()
    return accuracy


def check_purity(data: pd.DataFrame):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False


def classify_data(data: pd.DataFrame):
    label_columns = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_columns, return_counts=True)
    index = counts_unique_classes.argmax()
    classification = unique_classes[index]

    return classification


def get_potential_splits(data: pd.DataFrame, random_subspace: Optional[int]) -> dict:
    potential_splits = {}
    _, n_columns = data.shape
    column_indices = list(range(n_columns - 1))  # excluding the last column which is the label

    if random_subspace and random_subspace <= len(column_indices):
        column_indices = random.sample(population=column_indices, k=random_subspace)

    for column_index in column_indices:
        values = data[:, column_index]
        unique_values = np.unique(values)

        potential_splits[column_index] = unique_values

    return potential_splits


def split_data(data: pd.DataFrame, split_column: int, split_value: int, type_of_feature: str = "continuous"):
    split_column_values = data[:, split_column]

    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values > split_value]
    # feature is categorical   
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]

    return data_below, data_above


def calculate_entropy(data: pd.DataFrame) -> float:
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy


def calculate_overall_entropy(data_below: pd.DataFrame, data_above: pd.DataFrame) -> pd.DataFrame:
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy = (p_data_below * calculate_entropy(data_below)
                       + p_data_above * calculate_entropy(data_above))

    return overall_entropy


def determine_best_split(data: pd.DataFrame, potential_splits: dict) -> Tuple[int, int]:
    overall_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value
