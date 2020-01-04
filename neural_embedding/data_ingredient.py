from sacred import Ingredient
from data_loading import *


data_ingredient = Ingredient('dataset')


def prepare_inputs(df, selected_features_categorical, selected_features_numeric):
    inputs = {}
    for f in selected_features_categorical:
        inputs[f] = to_categorical(df[f])

    inputs['continuous'] = df[selected_features_numeric].as_matrix()
    return inputs


@data_ingredient.config
def cfg():
    filename = '../data'
    smote = False
    pca = False
    test_size = 0.25
    selected_features_numeric = [
        'gps_height',
        'latitude',
        'longitude',
        'population',
        'amount_tsh',
        'construction_year',
    ]

    selected_features_categorical = [
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

@data_ingredient.capture
def load_data(filename, test_size, smote, pca, selected_features_numeric, selected_features_categorical):
    all_features = selected_features_categorical + selected_features_numeric

    data = load_dataset(filename)
    experimentation_df, holdout_df = (data.pipe(data_cleaning)
                                    .pipe(numeric_groundtruth)
                                    .pipe(construction_year_feature)
                                    .pipe(feature_tsh_per_capita)
                                    .pipe(feature_removal, selected_features=selected_features)
                                    .pipe(split_data, test_size=test_size))
    train_df, test_df = split_data(experimentation_df)

    if pca:
        train_df, test_datasets = categorical_factorisation(data=train_df,test_datasets=[test_df,holdout_df])
        test_df = test_datasets[0]
        holdout_df = test_datasets[1]

    if smote:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(n_jobs=4, k=5)
        train_x_smote, train_y_smote = smote.fit_sample(train_df[all_features], train_df.status_group)
        train_inputs = prepare_inputs(pd.DataFrame(train_x_smote, columns=all_features))
        train_output = to_categorical(train_y_smote)
    else:
        train_inputs = prepare_inputs(train_df)
        train_output = to_categorical(train_df.status_group.as_matrix())


    test_inputs = prepare_inputs(test_df)
    holdout_inputs = prepare_inputs(holdout_df)
    test_output = to_categorical(test_df.status_group.as_matrix())
    holdout_output = to_categorical(holdout_df.status_group.as_matrix())

    return train_inputs, train_output, test_inputs, test_output, holdout_inputs, holdout_output