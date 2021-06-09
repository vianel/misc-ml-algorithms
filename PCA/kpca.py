import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def main():
    dt_hearth = pd.read_csv('./heart.csv')

    print(dt_hearth.head(5))

    dt_features = dt_hearth.drop(['target'], axis=1)
    dt_target = dt_hearth['target']

    dt_features = StandardScaler().fit_transform(dt_features)

    x_train, x_test, y_train, y_test = train_test_split(dt_features,
                                                        dt_target,
                                                        test_size=0.3,
                                                        random_state=42)

    print(x_train.shape)
    print(y_train.shape)

    kpca = KernelPCA(n_components=4, kernel='poly')

    kpca.fit(x_train)

    dt_train = kpca.transform(x_train)
    dt_test = kpca.transform(x_test)

    logistic = LogisticRegression(solver='lbfgs')

    logistic.fit(dt_train, y_train)

    print("KPCA acurracy:", logistic.score(dt_test, y_test))


if __name__ == '__main__':
    main()
