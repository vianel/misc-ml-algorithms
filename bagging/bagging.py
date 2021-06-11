import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    dt_heart = pd.read_csv('./heart.csv')

    print(dt_heart['target'].describe())

    x = dt_heart.drop(['target'], axis=1)
    y = dt_heart['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.35,
                                                        random_state=42)

    knn_class = KNeighborsClassifier()
    knn_class.fit(x_train, y_train)

    knn_pred = knn_class.predict(x_test)

    print("="*32)
    print(accuracy_score(knn_pred, y_test))

    bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(),
                                  n_estimators=50)

    bag_class.fit(x_train, y_train)
    bag_predict = bag_class.predict(x_test)
    print("="*32)
    print(accuracy_score(bag_predict, y_test))


if __name__ == "__main__":
    main()
