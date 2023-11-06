# Grid search
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from preprocess_train_data import unpickle_data, feature_extraction
import numpy as np

parameter_space = {
    'hidden_layer_sizes': [(100, 50, 25), (100, 70, 40, 10), (100, 45, 10), (70, 35, 17), (70, 52, 35, 17)],
    'max_iter': [5000],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05, 1e-3],
    'learning_rate': ['constant','adaptive'],
}

pkl_name = "config/car_colors"
width = 228

data = unpickle_data(pkl_name, width)

X = np.array(data["data"])
y = np.array(data["label"])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.35,
    shuffle=True,
    random_state=42,
)

X_train, X_test = feature_extraction(X_train, X_test, data)

mlp = MLPClassifier()

mlp_from_grid = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5, verbose=4)
mlp_from_grid.fit(X_train, y_train)

print('Best parameters found:\n', mlp_from_grid.best_params_)

# All results
means = mlp_from_grid.cv_results_['mean_test_score']
stds = mlp_from_grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, mlp_from_grid.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

y_true, y_pred = y_test , mlp_from_grid.predict(X_test)

from sklearn.metrics import classification_report
print('Results on the test set:')
print(classification_report(y_true, y_pred))