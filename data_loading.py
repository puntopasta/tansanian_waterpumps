import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from collections import Counter

def data_loading_pipeline(data_folder, test_size=0.25):
    '''
    Pipes all data loading and processing operations and returns the end result.
    :param data_folder: folder where data is located.
    :param test_size: size of the test split (fraction or absolute integer size)
    :return:
    '''
    data = load_dataset(data_folder)
    return (data.pipe(data_cleaning)
            .pipe(numeric_groundtruth)
            .pipe(fill_in_missing_construction_years)
            .pipe(feature_tsh_per_capita)
            .pipe(split_data, test_size=test_size)
     )


def load_dataset(data_folder):
    '''
    Loads all the waterpump data and concatenates everything into a single data frame
    :param data_folder:
    :return:
    '''
    labels_file = pd.read_csv(os.path.join(data_folder, 'water_pump_labels.csv')).set_index('id')
    set_file = pd.read_csv(os.path.join(data_folder,'water_pump_set.csv')).set_index('id')
    data = pd.concat([set_file, labels_file],axis=1)
    return data


def data_cleaning(data):
    '''
    Removes data with missing longitude.
    Removes columns with missing values.
    '''
    data = data.loc[data.longitude != 0]
    low_quality_columns = np.array(data.columns[data.isnull().sum() > 10])
    data = data.drop(low_quality_columns, axis=1)
    data.dropna(inplace=True)
    return data


def numeric_groundtruth(data):
    '''
    Re-maps ground truth column to numeric values according to schema:
    {'functional': 0, 'functional needs repair': 1, 'non functional': 2}
    '''
    mapping = {'functional': 0, 'functional needs repair': 1, 'non functional': 2}
    data.status_group = [mapping[item] for item in data.status_group]
    return data


def fill_in_missing_construction_years(data):
    '''
    Replaces all missing construction yaer entries with the mean construction year of the data set.
    '''
    mean_year = np.int(data[data.construction_year != 0].construction_year.mean())
    data[data.construction_year == 0] = mean_year
    return data


def feature_tsh_per_capita(data):
    '''
    Computes an additional feature "TSH per capita" and adds it as a column.
    :param data:
    :return:
    '''
    data['tsh_per_capita'] = data.tsh/data.population
    return data


def split_data(data, test_size=0.25, seed=42):
    '''
    Data splitting function.
    :param data: the complete dataframe.
    :param test_size: size of the training set (fraction or absolute integer size). Default is 0.25.
    :param seed: fix the randomisation for reproducibility. Default is the answer to life, the universe and everything.
    :return:
    '''
    import numpy as np
    np.random.seed(seed)
    train_df, test_df = train_test_split(data, test_size=test_size)
    print('Label distribution in training set: ', Counter(train_df.status_group))
    print('Label distribution in testing set: ', Counter(test_df.status_group))
    return train_df, test_df