import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn import preprocessing



def data_loading_pipeline(data_folder, selected_features = None, test_size=0.25):
    '''
    Pipes all data loading and processing operations and returns the end result.
    :param data_folder: folder where data is located.
    :param test_size: size of the test split (fraction or absolute integer size)
    :return:
    '''
    data = load_dataset(data_folder)
    return (data.pipe(data_cleaning)
            .pipe(numeric_groundtruth)
            .pipe(construction_year_feature)
            .pipe(feature_tsh_per_capita)
            .pipe(feature_removal, selected_features=selected_features)
            .pipe(split_data, test_size=test_size))



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



def construction_year_feature(data):
    '''
    Fills in missing contruction year.
    Also creates an "age_at_measurement" feature and fills it in.
    :param data
    :return: transformed data set.
    '''

    data.date_recorded = [x.year for x in data.date_recorded.astype('datetime64')]

    valid_year_data = data[data.construction_year != 0].copy()
    mean_construction_year = np.int(valid_year_data.construction_year.mean())
    mean_age = np.int(np.mean(valid_year_data.date_recorded-valid_year_data.construction_year))

    data['age_at_measurement'] = data.date_recorded - data.construction_year
    data['age_at_measurement'][data.construction_year == 0] = mean_age
    data.construction_year[data.construction_year == 0] = mean_construction_year

    return data


def feature_removal(data, selected_features = None):
    if selected_features == None:
        all_features = data.columns
    else:
        all_features = selected_features + ['status_group']
    return data[all_features]


def feature_tsh_per_capita(data):
    '''
    Computes an additional feature "TSH per capita" and adds it as a column.
    :param data:
    :return:
    '''
    data['tsh_per_capita'] = data.amount_tsh/data.population
    data['tsh_per_capita'][data['tsh_per_capita'].isnull()] = 0   # pop=0 and TSH=0, set feature to 0.
    data['tsh_per_capita'][data['tsh_per_capita'] == np.inf] = data['amount_tsh'][data['tsh_per_capita'] == np.inf]    # pop=0, TSH>0, assume pop=1 and set feature to TSH value.
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


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def categorical_factorisation(data, test_datasets):
    '''
    Uses matrix factorisation to arrive at numerical representation that captures most variance.
    :param data: data
    :return: data, with principal component representation of each categorical feature.
    '''
    from sklearn.decomposition import PCA

    for feature in data.columns:
        if data[feature].dtype == 'O':
            le = preprocessing.LabelEncoder()
            le.fit(data[feature])
            num = le.transform(data[feature])
            one_hot = to_categorical(num)
            arity = one_hot.shape[-1]
            if arity > 100:
                continue
            max_components = int(np.min([10, np.ceil(arity / 2)]))
            pca = PCA(n_components=max_components)
            components = pca.fit_transform(one_hot)
            component_names = ['{f}_{i}'.format(f=feature, i=i) for i in range(max_components)]
            new_features = pd.DataFrame(data=components, columns=component_names, index=data.index)
            data = pd.concat([data, new_features], axis=1)

            for i in range(len(test_datasets)):
                # now do the same for the test data, but re-apply the components from training.
                test_data = test_datasets[i]
                num = le.transform(test_data[feature])
                one_hot = to_categorical(num)
                components = pca.transform(one_hot)
                new_features = pd.DataFrame(data=components, columns=component_names, index=test_data.index)
                test_data = pd.concat([test_data, new_features], axis=1)
                test_datasets[i] = test_data
    return data, test_datasets

if __name__ == '__main__':
    selected_features = [
        'gps_height',
        'latitude',
        'longitude',
        'population',
        'amount_tsh',
        'age_at_measurement',
        'payment_type',
        'management_group',
        'quality_group',
        'region',
        'basin',
        'extraction_type_class',
        'quantity_group',
        'waterpoint_type_group',
        'source_type',
        'source_class'
    ]

    data_loading_pipeline(('data'), selected_features=selected_features)