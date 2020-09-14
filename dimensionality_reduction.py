import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



import seaborn as sns; sns.set()


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA, NMF



def task1c(wine):
    data = pd.read_csv("winequality-" + wine + ".csv", sep=';')
    X = data.drop('quality', axis=1)
    y = data['quality']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(X)
    pca_data = pca.transform(X)
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['label' + str(x) for x in range(1, len(per_var) + 1)]
    plt.bar(x=range(1, 12), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title(wine + ' wine PCA Variance')
    plt.show()
    pca_df = pd.DataFrame(pca_data, columns=labels)

    plt.scatter(pca_df.label1, pca_df.label2, edgecolors='black')
    plt.xlabel('label1 - {0}%'.format(per_var[0]))
    plt.ylabel('label2 - {0}%'.format(per_var[1]))
    plt.title(wine + ' wine PCA label1 and label2')
    plt.show()
    plt.scatter(pca_df.label1, pca_df.label3, edgecolors='black')
    plt.xlabel('label1 - {0}%'.format(per_var[0]))
    plt.ylabel('label3 - {0}%'.format(per_var[2]))
    plt.title(wine + ' wine PCA label1 and label3')
    plt.show()
    plt.scatter(pca_df.label2, pca_df.label3, edgecolors='black')
    plt.xlabel('label2 - {0}%'.format(per_var[1]))
    plt.ylabel('label3 - {0}%'.format(per_var[2]))
    plt.title(wine + ' wine PCA label2 and label3')
    plt.show()


    print("correlation between label1 and quality: " + str(pca_df.label1.corr(y)))
    print("correlation between label2 and quality: " + str(pca_df.label2.corr(y)))
    print("correlation between label3 and quality: " + str(pca_df.label3.corr(y)))

    plt.clf()

    ica = FastICA(n_components=3)
    ica.fit(X)
    ica_data = ica.transform(X)
    labels = ['label' + str(x) for x in range(1, 4)]
    ica_df = pd.DataFrame(ica_data, columns=labels)

    plt.scatter(ica_df.label1, ica_df.label2, edgecolors='black')
    plt.xlabel('label1')
    plt.ylabel('label2')
    plt.title(wine + ' wine ICA label1 and label2')
    plt.show()
    plt.scatter(ica_df.label1, ica_df.label3, edgecolors='black')
    plt.xlabel('label1')
    plt.ylabel('label3')
    plt.title(wine + ' wine ICA label1 and label3')
    plt.show()
    plt.scatter(ica_df.label2, ica_df.label3, edgecolors='black')
    plt.xlabel('label2')
    plt.ylabel('label3')
    plt.title(wine + ' wine ICA label2 and label3')
    plt.show()

    print("correlation between label1 and quality: " + str(ica_df.label1.corr(y)))
    print("correlation between label2 and quality: " + str(ica_df.label2.corr(y)))
    print("correlation between label3 and quality: " + str(ica_df.label3.corr(y)))

    plt.clf()
    X = data.drop('quality', axis=1)
    y = data['quality']
    nmf = NMF(n_components=3, max_iter=10000)
    nmf.fit(X)
    nmf_data = nmf.transform(X)
    nmf_df = pd.DataFrame(nmf_data, columns=labels)

    plt.scatter(nmf_df.label1, nmf_df.label2, edgecolors='black')
    plt.xlabel('label1')
    plt.ylabel('label2')
    plt.title(wine + ' wine NMF label1 and label2')
    plt.show()
    plt.scatter(nmf_df.label1, nmf_df.label3, edgecolors='black')
    plt.xlabel('label1')
    plt.ylabel('label3')
    plt.title(wine + ' wine NMF label1 and label3')
    plt.show()
    plt.scatter(nmf_df.label2, nmf_df.label3, edgecolors='black')
    plt.xlabel('label2')
    plt.ylabel('label3')
    plt.title(wine + ' wine NMF label2 and label3')
    plt.show()

    print("correlation between label1 and quality: " + str(nmf_df.label1.corr(y)))
    print("correlation between label2 and quality: " + str(nmf_df.label2.corr(y)))
    print("correlation between label3 and quality: " + str(nmf_df.label3.corr(y)))


    print("end")