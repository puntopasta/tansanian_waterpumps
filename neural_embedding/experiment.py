import warnings
import numpy as np
from sacred.stflow import LogFileWriter
from sacred import Experiment
from sacred.observers import FileStorageObserver
from keras.models import Model
from keras import layers as L
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from neural_embedding.data_ingredient import data_ingredient
warnings.filterwarnings('ignore')


ex = Experiment("my experiment", ingredients=[data_ingredient])
ex.observers.append(FileStorageObserver('my_runs'))


@ex.config
def config():
    return None


@ex.automain
@LogFileWriter(ex)
def my_main(selected_features_categorical):
    train_inputs, train_output, test_inputs, test_output, holdout_inputs, holdout_output = load_data()
    print('Input shapes: ')
    for k in train_inputs.keys():
        print(train_inputs[k].shape)

    print('Output shape: {}'.format(train_output.shape))

    input_layers = []
    input_name_orders = []
    concat_layers = []
    for k in selected_features_categorical:
        input_layers.append(L.Input(shape=(train_inputs[k].shape[-1],), name=k))
        embed = L.Embedding(input_dim=train_inputs[k].shape[-1],
                            output_dim=int(np.ceil(train_inputs[k].shape[-1]/2)), name='{}_embed'.format(k))(input_layers[-1])
        embed = L.Flatten()(embed)
        concat_layers.append(embed)

        input_name_orders.append(k)

    input_layers.append(L.Input(shape=(train_inputs['continuous'].shape[-1],), name='continuous'))
    input_name_orders.append('continuous')
    concat_layers.append(L.Dense(32, name='continuous_dense', activation='sigmoid')(input_layers[-1]))

    latent = L.concatenate(concat_layers)
    latent = L.Dropout(0.5)(latent)
    latent = L.Dense(15, activation='sigmoid')(latent)
    output = L.Dense(train_output.shape[-1], activation='softmax', name='decision')(latent)

    model = Model(inputs=input_layers, outputs=output, name='neural_embedder')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    model.summary()

    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

    callbacks = [EarlyStopping(patience=50),
                 TensorBoard(log_dir='../logs',embeddings_freq=20),
                 ReduceLROnPlateau(factor=0.5,patience=20)]

    class_weights = {}
    for i in [0, 1, 2]:
        class_weights[i] = np.square(1 - (np.sum(np.argmax(train_output, -1) == i) / len(train_output)))

    model.fit(x=[train_inputs[k] for k in input_name_orders], y=train_output, epochs=5000,
              validation_data=([test_inputs[k] for k in input_name_orders], test_output),
              callbacks=callbacks, batch_size=64
              , class_weight=class_weights)


    holdout_pred = np.argmax(model.predict(holdout_inputs),-1)
    holdout_ref = np.argmax(holdout_output,-1)


    _run.info['test_confusion_matrix'] = confusion_matrix(holdout_ref, holdout_pred)
    _run.info['test_kappa'] = cohen_kappa_score(holdout_ref, holdout_pred)
    _run.info['test_accuracy'] = accuracy_score(holdout_ref, holdout_pred)
    _run.info['holdout_confusion_matrix'] = confusion_matrix(holdout_ref, holdout_pred)
    _run.info['holdout_kappa'] = cohen_kappa_score(holdout_ref, holdout_pred)
    _run.info['holdout_accuracy'] = accuracy_score(holdout_ref, holdout_pred)

    return _run.info['test_kappa'] # use test performance as selection mechanism.


