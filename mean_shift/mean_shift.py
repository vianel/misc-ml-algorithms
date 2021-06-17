import pandas as pd

from sklearn.cluster import MeanShift

if __name__ == '__main__':

    dataset = pd.read_csv('candy.csv')
    print(dataset.head(5))

    x = dataset.drop('competitorname', axis=1)

    meanShift = MeanShift().fit(x)

    print('='*32)
    print(max(meanShift.labels_))
    print('='*32)
    print(meanShift.labels_)
    print('='*32)
    print(meanShift.cluster_centers_)

    dataset['meanshift'] = meanShift.labels_
    print('='*32)
    print(dataset.head(10))
