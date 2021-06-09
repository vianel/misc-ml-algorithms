import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def main():
    dataset = pd.read_csv("./whr2017.csv")
    print(dataset.describe())

    x = dataset[['gdp', 'family', 'lifexp', 'freedom',
                 'corruption', 'generosity', 'dystopia']]

    y = dataset[['score']]

    print(x.shape, y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    modelLinear = LinearRegression().fit(x_train, y_train)
    y_predict_linear = modelLinear.predict(x_test)

    lasso = Lasso(alpha=0.02).fit(x_train, y_train)
    y_predict_lasso = lasso.predict(x_test)

    ridge = Ridge(alpha=1).fit(x_train, y_train)
    y_predict_ridge = ridge.predict(x_test)

    elastic = ElasticNet().fit(x_train, y_train)
    y_predict_elastic = elastic.predict(x_test)

    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print("Linear Loss:", linear_loss)

    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print("Lasso loss:", lasso_loss)

    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print("Ridge loss:", ridge_loss)

    elastic_loss = mean_squared_error(y_test, y_predict_elastic)
    print("Elastic loss:", elastic_loss)

    print("="*32)
    print("Coef lasso", lasso.coef_)
    print("Coef Ridge", ridge.coef_)


if __name__ == '__main__':
    main()
