import warnings
from sacred.stflow import LogFileWriter
from sacred import Experiment
from sacred.observers import MongoObserver
from keras.models import Model
from keras import layers as L
from data_loading import *
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
warnings.filterwarnings('ignore')


def prepare_inputs(df, selected_features_categorical, selected_features_numeric):
    inputs = {}
    for f in selected_features_categorical:
        inputs[f] = to_categorical(df[f])

    inputs['continuous'] = df[selected_features_numeric].as_matrix()
    return inputs

ex = Experiment("my experiment")
ex.observers.append(MongoObserver())


@ex.config
def config():
    activation = 'sigmoid'
    dropout = 0
    n_latent = 16
    n_feature_selector = 32
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


@ex.automain
@LogFileWriter(ex)
def my_main(_run, activation, dropout, n_latent, n_feature_selector, filename, test_size, smote, pca, selected_features_numeric, selected_features_categorical):

    # DATA LOADING

    all_features = selected_features_categorical + selected_features_numeric

    data = load_dataset(filename)
    experimentation_df, holdout_df = (data.pipe(data_cleaning)
                                      .pipe(numeric_groundtruth)
                                      .pipe(construction_year_feature)
                                      .pipe(feature_tsh_per_capita)
                                      .pipe(feature_removal, selected_features=all_features)
                                      .pipe(categorical_encoding)
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
        train_inputs = prepare_inputs(pd.DataFrame(train_x_smote, columns=all_features),
                                      selected_features_categorical=selected_features_categorical,
                                      selected_features_numeric=selected_features_numeric)
        train_output = to_categorical(train_y_smote)
    else:
        train_inputs = prepare_inputs(train_df,
                                      selected_features_categorical=selected_features_categorical,
                                      selected_features_numeric=selected_features_numeric)
        train_output = to_categorical(train_df.status_group.as_matrix())


    test_inputs = prepare_inputs(test_df,
                                      selected_features_categorical=selected_features_categorical,
                                      selected_features_numeric=selected_features_numeric)
    holdout_inputs = prepare_inputs(holdout_df,
                                      selected_features_categorical=selected_features_categorical,
                                      selected_features_numeric=selected_features_numeric)
    test_output = to_categorical(test_df.status_group.as_matrix())
    holdout_output = to_categorical(holdout_df.status_group.as_matrix())


    print('Input shapes: ')
    for k in train_inputs.keys():
        print(train_inputs[k].shape)

    print('Output shape: {}'.format(train_output.shape))

    # MODEL BUILDING
    input_layers = []
    input_name_orders = []
    embedding_layer_names = []
    embedding_layer_data = []
    concat_layers = []
    for k in selected_features_categorical:
        input_layers.append(L.Input(shape=(train_inputs[k].shape[-1],), name=k))
        layer_name = '{}_embed'.format(k)
        embed = L.Embedding(input_dim=train_inputs[k].shape[-1],
                            output_dim=int(np.ceil(train_inputs[k].shape[-1]/2)), name=layer_name)(input_layers[-1])
        embed = L.Flatten()(embed)
        embedding_layer_names.append(layer_name)
        embedding_layer_data.append(train_inputs[k])
        concat_layers.append(embed)

        input_name_orders.append(k)

    input_layers.append(L.Input(shape=(train_inputs['continuous'].shape[-1],), name='continuous'))
    input_name_orders.append('continuous')
    concat_layers.append(L.Dense(n_feature_selector, name='continuous_dense', activation=activation)(input_layers[-1]))

    latent = L.concatenate(concat_layers)
    if dropout != 0:
        latent = L.Dropout(dropout)(latent)
    latent = L.Dense(n_latent, activation=activation)(latent)
    output = L.Dense(train_output.shape[-1], activation='softmax', name='decision')(latent)

    model = Model(inputs=input_layers, outputs=output, name='neural_embedder')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    model.summary()

    # MODEL TRAINING
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

    callbacks = [EarlyStopping(patience=50),
                 TensorBoard(log_dir='../logs'), # , embeddings_freq=20,embeddings_layer_names=input_name_orders, embeddings_data=[train_inputs[k] for k in input_name_orders]
                 ReduceLROnPlateau(factor=0.5, patience=20)]

    class_weights = {}
    for i in [0, 1, 2]:
        class_weights[i] = np.square(1 - (np.sum(np.argmax(train_output, -1) == i) / len(train_output)))

    model.fit(x=[train_inputs[k] for k in input_name_orders], y=train_output, epochs=5000,
              validation_data=([test_inputs[k] for k in input_name_orders], test_output),
              callbacks=callbacks, batch_size=64
              , class_weight=class_weights)

    # MODEL EVALUATION

    holdout_pred = np.argmax(model.predict(holdout_inputs),-1)
    holdout_ref = np.argmax(holdout_output,-1)


    _run.info['test_confusion_matrix'] = confusion_matrix(holdout_ref, holdout_pred)
    _run.info['test_kappa'] = float(cohen_kappa_score(holdout_ref, holdout_pred))
    _run.info['test_accuracy'] = float(accuracy_score(holdout_ref, holdout_pred))
    _run.info['holdout_confusion_matrix'] = confusion_matrix(holdout_ref, holdout_pred)
    _run.info['holdout_kappa'] = float(cohen_kappa_score(holdout_ref, holdout_pred))
    _run.info['holdout_accuracy'] = float(accuracy_score(holdout_ref, holdout_pred))

    return _run.info['test_kappa'] # use test performance as selection mechanism.


