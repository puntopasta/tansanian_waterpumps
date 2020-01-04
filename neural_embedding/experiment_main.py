import sys
sys.path.append('.')
from itertools import product

from neural_embedding.experiment import ex

for n_latent, n_feature_selector, activation, dropout in product([8,16,32,64],[8,16,32,64],['sigmoid','linear','relu'], [0,0.5,0.8]):
    ex.run(config_updates=
           {
               'filename':'data',
               'n_latent':n_latent,
               'activation': activation,
               'n_feature_selector':n_feature_selector,
               'dropout':dropout
           })

