    # # Grid search
    # from sklearn.model_selection import GridSearchCV

    # parameter_space = {
    #     'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,), (100, 100, 100)],
    #     'activation': ['tanh', 'relu'],
    #     'solver': ['sgd', 'adam'],
    #     'alpha': [0.0001, 0.05, 1e-3, 5e-3, 0.5e-3],
    #     'learning_rate': ['constant','adaptive'],
    # }

    # clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    # clf.fit(X_train, y_train)

    # print('Best parameters found:\n', clf.best_params_)

    # # All results
    # means = clf.cv_results_['mean_test_score']
    # stds = clf.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    # y_true, y_pred = y_test , clf.predict(X_test)

    # from sklearn.metrics import classification_report
    # print('Results on the test set:')
    # print(classification_report(y_true, y_pred))