import pandas as pd

# This is useful for computer with lower resources
from sklearn.cluster import MiniBatchKMeans


def main():
    dataset = pd.read_csv('candy.csv')
    print(dataset.head(5))

    x = dataset.drop('competitorname', axis=1)

    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(x)

    print('Total centers ', len(kmeans.cluster_centers_))
    print('='*64)
    result = kmeans.predict(x)
    print(result)

    dataset['group'] = result

    print(dataset.head(10))


if __name__ == '__main__':
    main()
