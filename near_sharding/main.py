from calendar import c
from cgi import test
import warnings
from sklearn.cluster import KMeans
import random
import numpy as np
import pandas as pd
import scipy.spatial.distance as metric
import math
import sklearn.datasets as datasets
import time
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import normalize

np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")

# datasets
iris = datasets.load_iris()
wine = datasets.load_wine()

# Helpers


def calcSSE(data, centroids):
    sum = 0
    for i in data:
        distance = math.inf
        for k in centroids:
            if euc(i, k) < distance:
                distance = euc(i, k)
        sum += distance**2

    return sum / len(data)


def euc(A, B):
    # Call to scipy with vector parameters

    return metric.euclidean(A, B)


# Metots


def rand_cent(ds, k):
    # Number of columns in dataset
    n = np.shape(ds)[1]

    # The centroids
    centroids = np.mat(np.zeros((k, n)))

    # Create random centroids
    for j in range(n):

        min_j = min(ds[:, j])
        range_j = float(max(ds[:, j]) - min_j)
        centroids[:, j] = min_j + range_j * np.random.rand(k, 1)

    # Return centroids as numpy array
    return centroids


def random_datapoints(ds, k):
    index_list = random.sample(range(1, len(ds)), k)
    centroids = ds[index_list]
    return centroids


def naive_sharding(ds, k):
    n = np.shape(ds)[1]

    m = np.shape(ds)[0]

    centroids = np.mat(np.zeros((k, n)))

    composite = np.sum(ds, axis=1)
    composite = np.reshape(composite, (len(ds), 1))

    ds = np.append(composite, ds, axis=1)

    ds.sort(axis=0)
    step = math.floor(m / k)

    vfunc = np.vectorize(_get_mean)

    for j in range(k):
        if j == k - 1:
            centroids[j:] = vfunc(np.sum(ds[j * step :, 1:], axis=0), step)
        else:
            centroids[j:] = vfunc(
                np.sum(ds[j * step : (j + 1) * step, 1:], axis=0), step
            )

    return centroids


def mean_sharding(ds, k):
    n = np.shape(ds)[1]

    m = np.shape(ds)[0]

    centroids = np.mat(np.zeros((k, n)))

    composite = np.mean(ds, axis=1)
    composite = np.reshape(composite, (len(ds), 1))

    ds = np.append(composite, ds, axis=1)

    # ds = ds[ds[:, 0].argsort(kind="mergesort")]
    ds.sort(axis=0)

    step = math.floor(m / k)

    vfunc = np.vectorize(_get_mean)

    for j in range(k):
        if j == k - 1:
            centroids[j:] = vfunc(np.sum(ds[j * step :, 1:], axis=0), step)
        else:
            centroids[j:] = vfunc(
                np.sum(ds[j * step : (j + 1) * step, 1:], axis=0), step
            )

    return centroids


def median_sharding(ds, k):
    n = np.shape(ds)[1]

    m = np.shape(ds)[0]

    centroids = np.mat(np.zeros((k, n)))

    composite = np.median(ds, axis=1)
    composite = np.reshape(composite, (len(ds), 1))

    ds = np.append(composite, ds, axis=1)

    # ds = ds[ds[:, 0].argsort()]
    ds.sort(axis=0)

    step = math.floor(m / k)

    vfunc = np.vectorize(_get_mean)

    for j in range(k):
        if j == k - 1:
            centroids[j:] = vfunc(np.sum(ds[j * step :, 1:], axis=0), step)
        else:
            centroids[j:] = vfunc(
                np.sum(ds[j * step : (j + 1) * step, 1:], axis=0), step
            )

    return centroids


def split_arr(ds, threshold, j):
    if np.size(ds) == 0:
        return None
    min_val = ds[0, 0]

    k = 0
    for i in range(len(ds)):
        if ds[k, 0] - min_val <= threshold:
            # print(k)
            k += 1
        else:
            break

    return [j + k, ds[0:k, :]]


def minmaxsharding(ds, k):

    n = np.shape(ds)[1]

    centroids = np.mat(np.zeros((k, n)))

    composite = np.sum(ds, axis=1)

    composite = np.reshape(composite, (len(ds), 1))

    ds = np.append(composite, ds, axis=1)
    # print(ds)

    # ds = ds[ds[:, 0].argsort()]
    ds.sort(axis=0)

    # print(ds)
    ds_range = np.max(ds[:, 0]) - np.min(ds[:, 0])

    threshold = ds_range / k
    prev_arr = split_arr(ds, threshold, 0)

    for j in range(k):
        # print(prev_arr[1])
        centroids[j, :] = np.sum(prev_arr[1][:, 1:], axis=0) / np.shape(prev_arr[1])[0]
        # print(centroids)

        prev_arr = split_arr(ds[prev_arr[0] :, :], threshold, prev_arr[0])
        # print("done")

    return centroids


def l_inf(datas):
    # datas bir numpy arrayi. Parametre olarak np array verilmeli veya alttaki yorum satıtındaki kod çalıştırılmalı:
    # datas=datas.to_numpy()
    num = len(datas[0])
    for i in range(num):
        m = max(datas[:, i])
        datas[:, i] = datas[:, i] / m
    return datas


def norm_sharding(ds, k):

    n = np.shape(ds)[1]
    m = np.shape(ds)[0]

    centroids = np.mat(np.zeros((k, n)))

    normds = ds.copy()
    normds = normalize(normds, axis=0, norm="max")
    composite = np.sum(normds, axis=1)
    composite = np.reshape(composite, (len(ds), 1))

    ds = np.append(composite, ds, axis=1)

    ds.sort(axis=0)
    step = math.floor(m / k)

    vfunc = np.vectorize(_get_mean)

    for j in range(k):
        if j == k - 1:
            centroids[j:] = vfunc(np.sum(ds[j * step :, 1:], axis=0), step)
        else:
            centroids[j:] = vfunc(
                np.sum(ds[j * step : (j + 1) * step, 1:], axis=0), step
            )

    return centroids





def near_sharding(ds, k):



    n = np.shape(ds)[1]

    m = np.shape(ds)[0]

    centroids = np.mat(np.zeros((k, n)))

    composite = np.sum(ds, axis=1)

    composite = np.reshape(composite, (len(ds), 1))

    ds = np.append(composite, ds, axis=1)
    # print(ds)

    # ds = ds[ds[:, 0].argsort()]
    ds.sort(axis=0)

    breakPoints = np.zeros((k - 1), dtype=int)

    partSize = int(math.floor(m / k))

    for i in range(k - 1):
        breakPoints[i] = partSize * (i + 1)
    
    step = 0 
    while step < k - 1:
        
        prev_point = 0
        next_point = m - 1
        break_point = breakPoints[step]
        if step != 0:
            prev_point = breakPoints[step - 1]
        if step < k - 2:
            next_point = breakPoints[step + 1]
        
        current_mean = np.sum(ds[prev_point:(break_point - 1), 1:], axis=0) / ((break_point - prev_point)) 
        next_mean = np.sum(ds[break_point:next_point, 1:], axis=0) / ((next_point - break_point)) 
        
        current_distance = euc(current_mean, ds[break_point, 1:])
        next_distance = euc(next_mean, ds[break_point, 1:])
        if next_distance > current_distance:
            breakPoints[step] = breakPoints[step] + 1
            step = step + 1
        else:
            breakPoints[step] = breakPoints[step] - 1           

    for j in range(k):
        if j == 0:
            centroids[j:] = np.sum(ds[0:breakPoints[0] - 1, 1:], axis=0) / breakPoints[j] - 1
        elif j == k - 1: 
            centroids[j:] = np.sum(ds[breakPoints[j - 1]:, 1:], axis=0) / (m - breakPoints[j - 1]) 
        else:
            centroids[j:] = np.sum(ds[breakPoints[j - 1]:breakPoints[j] - 1, 1:], axis=0) / (breakPoints[j] - breakPoints[j - 1] - 1) 

    return centroids


def kmeans(ds, k, cent_method):

    global timer_start
    global timer_end
    global total_timer_end
    return_object = {}
    global cents
    global sse
    global iters
    cents = []

    if cent_method == "random":
        timer_start = time.perf_counter()
        cents = rand_cent(ds, k)
        timer_end = time.perf_counter()
    elif cent_method == "random_datapoints":
        timer_start = time.perf_counter()
        cents = random_datapoints(ds, k)
        timer_end = time.perf_counter()
    elif cent_method == "naive":
        timer_start = time.perf_counter()
        cents = naive_sharding(ds, k)
        timer_end = time.perf_counter()
    elif cent_method == "mean":
        timer_start = time.perf_counter()
        cents = mean_sharding(ds, k)
        timer_end = time.perf_counter()
    elif cent_method == "median":
        timer_start = time.perf_counter()
        cents = median_sharding(ds, k)
        timer_end = time.perf_counter()
    elif cent_method == "minmax":
        timer_start = time.perf_counter()
        cents = minmaxsharding(ds, k)
        timer_end = time.perf_counter()
    elif cent_method == "norm_sharding":
        timer_start = time.perf_counter()
        cents = norm_sharding(ds, k)
        timer_end = time.perf_counter()
    elif cent_method == "binary_method":
        timer_start = time.perf_counter()
        cents = binary_method(ds, k)
        timer_end = time.perf_counter()
    elif cent_method == "near_sharding":
        timer_start = time.perf_counter()
        cents = near_sharding(ds, k)
        timer_end = time.perf_counter()

    km = KMeans(n_clusters=k, init=cents).fit(ds)
    total_timer_end = time.perf_counter()
    iters = km.n_iter_
    cents = km.cluster_centers_
    sse = km.inertia_
    return_object["cents"] = cents
    return_object["time"] = timer_end - timer_start
    return_object["total-time"] = total_timer_end - timer_start
    return_object["sse"] = sse / len(ds)
    return_object["type"] = cent_method
    return_object["iter"] = iters
    return return_object


def _get_mean(sums, step):
    return sums / step


def printResult(datas):
    print(
        "{:<30} {:<30} {:<30} {:<30} {:<30}".format(
            "Type", "Time", "SSE", "Total Time", "Iter"
        )
    )
    print("-" * 120)
    for d in datas:
        print(
            "{:<30} {:<30} {:<30} {:<30} {:<30}".format(
                d["type"], d["time"], d["sse"], d["total-time"], d["iter"]
            )
        )


iris_ds = iris.data


# df = pd.read_csv("./near_sharding/ruspini.csv")  # np arrayi
# df = df.to_numpy()

df = wine.data
df = df.astype(float)
df = l_inf(df)



def binary_method(ds, k):
    if k == 2:
        centroids = np.array([ds.min(axis=0), ds.max(axis=0)])

    else:
        min_cent = ds.min(axis=0)

        max_cent = ds.max(axis=0)

        centroids = np.array([min_cent, max_cent])

        diff = (max_cent - min_cent) / (k - 1)

        for i in range(k - 2):
            centroids = np.append(centroids, [min_cent + (i + 1) * diff], axis=0)

    return centroids


methods = [
    "random",
    "minmax",
    "median",
    "mean",
    "naive",
    "random_datapoints",
    "norm_sharding",
    "binary_method",
    "near_sharding",
]
values = [[0 for j in range(5)] for i in range(len(methods))]
repetition = 300
for j in range(len(methods)):
    for i in range(repetition):
        values[j][0] = methods[j]
        a = kmeans(df, 3, methods[j])
        values[j][1] += a["time"]
        values[j][2] += a["sse"]
        values[j][3] += a["total-time"]
        values[j][4] += a["iter"]
    for k in range(4):
        values[j][k + 1] = values[j][k + 1] / repetition

print(
    "{:<30} {:<30} {:<30} {:<30} {:<30}".format(
        "Type", "Time", "SSE", "Total Time", "Iter"
    )
)
for i in range(len(values)):
    print(
        "{:<30} {:<30} {:<30} {:<30} {:<30}".format(
            values[i][0], values[i][1], values[i][2], values[i][3], values[i][4]
        )
    )
