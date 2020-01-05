# tansanian_waterpumps
exploration and analysis of the tansanian waterpump maintenance dataset

# Dependencies
Python 3.5 with packages:
numpy keras pandas sacred scikit-learn seaborn tensorflow imbalanced-learn (latest versions should work)

# Random forest model
The random forest model is created and evaluated in notebooks in the random_forest directory. Two notebooks exist. In one of them, the categorical data is encoded as numeric variables. In the other, the categorical data first undergoes matrix factorisation and the principal components are used as features.

# Neural embedding model
The neural embedding model uses the Sacred experiment tracking environment to track different hyper-parameters. Sacred stored all information in a mongo database (which should be installed). Visualising the results can be done with Omniboard.

A jupyter notebook example is also included.

https://sacred.readthedocs.io/en/stable/

https://vivekratnavel.github.io/omniboard/#/quick-start
