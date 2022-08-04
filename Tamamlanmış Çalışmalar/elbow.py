from numpy import number
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def calculateKWithElbow(data):
    wcss = []
    for i in range(1, 20):
        kmeans = KMeans(i)
        kmeans.fit(data)
        wcss_iter = kmeans.inertia_
        wcss.append(wcss_iter)
    number_clusters = range(1, 20)
    plt.plot(number_clusters, wcss)
    plt.title("Elbow")
    plt.xlabel('Number Of Methods')
    plt.ylabel('WCSS')
    plt.show()
