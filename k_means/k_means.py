from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


def main():
    iris = datasets.load_iris()

    y_iris = iris.target

    x = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Witdth',
                                         'Petal Length', 'Petal Width'])

    print(x.head(5))

    plt.scatter(x['Petal Length'], x['Petal Width'], c='blue')
    plt.xlabel('Petal Length', fontsize=10)
    plt.ylabel('Petal Width', fontsize=10)
    plt.show()

    model = KMeans(n_clusters=3, max_iter=2000)
    model.fit(x)

    y_labels = model.labels_
    print('Labels, ', y_labels)

    y_kmeans = model.predict(x)

    print('Predictions ', y_kmeans)

    accuracy = metrics.adjusted_rand_score(y_iris, y_kmeans)
    print('Accuracy {:.2%}'.format(accuracy))

    plt.scatter(x['Petal Length'], x['Petal Width'], c=y_kmeans, s=30)
    plt.xlabel('Petal Length', fontsize=10)
    plt.ylabel('Petal Width', fontsize=10)
    plt.show()


if __name__ == '__main__':
    main()
