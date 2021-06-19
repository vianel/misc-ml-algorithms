import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


if __name__ == '__main__':
    dataset = pd.read_csv('happiness.csv')
    print(dataset.head(5))

    x = dataset.drop(['country', 'rank', 'score'], axis=1)
    y = dataset['score']

    regressor = RandomForestRegressor()

    parameters = {
        'n_estimators': range(4, 16),
        'criterion': ['mse', 'mae'],
        'max_depth': range(2, 11)
    }

    rand_est = RandomizedSearchCV(regressor, parameters, n_iter=10, cv=3,
                                  scoring='neg_mean_absolute_error').fit(x, y)

    print(rand_est.best_estimator_)
    print(rand_est.best_params_)
    print(rand_est.predict(x.loc[[0]]))
