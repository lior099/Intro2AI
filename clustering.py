import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator



import seaborn as sns; sns.set()


from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture




def task1b(wine):
    data = pd.read_csv("winequality-" + wine + ".csv", sep=';')
    X = data.drop('quality', axis=1)
    y = data['quality']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    kmeans_args = {"n_init": 10, "max_iter": 300, "random_state": 0}
    sse = []
    silhouette_coefficients = []
    max_k = 20
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, **kmeans_args)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
        if k > 1:
            score = silhouette_score(X, kmeans.labels_)
            silhouette_coefficients.append(score)
    plt.style.use("fivethirtyeight")
    plt.plot(range(1, max_k + 1), sse)
    plt.xticks(range(1, max_k + 1))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.title(wine + ' wine SSE')
    plt.show()
    kl = KneeLocator(range(1, max_k + 1), sse, curve = "convex", direction = "decreasing")
    print(kl.elbow)
    k = kl.elbow
    plt.style.use("fivethirtyeight")
    plt.plot(range(2, max_k + 1), silhouette_coefficients)
    plt.xticks(range(2, max_k + 1))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.title(wine + ' wine Silhouette Coefficient')
    plt.show()

    kmeans_means = []
    kmeans = KMeans(n_clusters=k, **kmeans_args)
    kmeans.fit(X)
    clusters_quality = [[] for i in range(k)]
    for i in range(len(y)):
        clusters_quality[kmeans.labels_[i]].append(y[i])
    for i in range(k):
        print("Mean quality of cluster " + str(i) + " is " + str(sum(clusters_quality[i]) / len(clusters_quality[i])))
        kmeans_means.append(round(sum(clusters_quality[i]) / len(clusters_quality[i]), 2))


    ac_means = []
    ac = AgglomerativeClustering(n_clusters=k)
    ac.fit(X)
    clusters_quality = [[] for i in range(k)]
    for i in range(len(y)):
        clusters_quality[ac.labels_[i]].append(y[i])
    for i in range(k):
        print("Mean quality of cluster " + str(i) + " is " + str(sum(clusters_quality[i]) / len(clusters_quality[i])))
        ac_means.append(round(sum(clusters_quality[i]) / len(clusters_quality[i]), 2))

    gmm_means = []
    gmm = GaussianMixture(n_components=k, **kmeans_args)
    gmm.fit(X)
    labels = gmm.predict(X)
    clusters_quality = [[] for i in range(k)]
    for i in range(len(y)):
        clusters_quality[labels[i]].append(y[i])
    for i in range(k):
        print("GaussianMixture - Mean quality of cluster " + str(i) + " is " + str(sum(clusters_quality[i]) / len(clusters_quality[i])))
        gmm_means.append(round(sum(clusters_quality[i]) / len(clusters_quality[i]), 2))

    cell_text = [sorted(kmeans_means), sorted(ac_means), sorted(gmm_means)]
    row_labels = ["K-Means", "Hierarchical Clustering", "GMM"]

    plt.table(cellText=cell_text, rowLabels=row_labels, loc="center", colWidths=k*[0.1])
    plt.title(wine + ' wine Mean Quality of Each of '+str(k)+' cluster')
    plt.axis("off")
    plt.grid(False)
    plt.show()