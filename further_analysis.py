import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



import seaborn as sns; sns.set()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import scipy

def task3(wine):
    data = pd.read_csv("winequality-" + wine + ".csv", sep=';')
    edges_dict = {}
    features_scores = []
    for feature in data.columns:
        print("Now doing "+feature)
        X = data.drop(feature, axis=1)
        y = data[feature]
        feature_list = list(X.columns)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        rf = RandomForestRegressor(n_estimators=30, random_state=0)
        rf.fit(X_train, y_train)
        print("score: ", rf.score(X_test, y_test))
        features_scores.append(rf.score(X_test, y_test))
        importances = list(rf.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        [print('Variable: {:24} Importance: {}'.format(*pair)) for pair in feature_importances]

        importances = [importance for feature, importance in feature_importances]
        feature_list = [feature for feature, importance in feature_importances]
        plt.bar(x=range(1, 12), height=importances, tick_label=feature_list)
        plt.ylabel('ylabel')
        plt.xlabel('xlabel')
        plt.title(wine + ' wine '+feature+' Feature Importances')
        plt.show()
        used_features = []
        scores = []
        for f in feature_list:
            used_features.append(f)
            used_X = data[used_features]
            X_train, X_test, y_train, y_test = train_test_split(used_X, y, test_size=0.25, random_state=0)
            rf = RandomForestRegressor(n_estimators=5, random_state=0)
            rf.fit(X_train, y_train)
            scores.append(rf.score(X_test, y_test))
        plt.style.use("fivethirtyeight")
        plt.plot(range(1, 12), scores)
        plt.xticks(range(1, 12))
        plt.xlabel("Number of Features")
        plt.ylabel("R2 score")
        plt.title(wine + ' wine ' + feature + ' R2 score')
        plt.show()

        edges_dict[feature] = feature_list[:scores.index(max(scores))+1]
        print()

    cell_text = [[round(score, 2) for score in features_scores]]
    col_labels = list(data.columns)
    plt.clf()
    col_widths = [len(a) / 120 for a in col_labels]
    col_widths[8] *= 3
    the_table = plt.table(cellText=cell_text, colLabels=col_labels, loc="center", colWidths=col_widths)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(13)
    plt.title(wine + ' wine R2 Score of Each of 12 attributes')
    plt.axis("off")
    plt.grid(False)
    plt.show()

    G = nx.DiGraph()
    edges = []
    feature_count_dict = {}
    for feature, learning_features in edges_dict.items():
        for learning_feature in learning_features:
            if learning_feature not in feature_count_dict.keys():
                feature_count_dict[learning_feature] = 0
            feature_count_dict[learning_feature] += 1

    quantity_add = {feature: str(feature_count_dict[feature]) + "\n" + feature for feature in edges_dict.keys()}
    for feature, learning_features in edges_dict.items():
        for learning_feature in learning_features:
            edges.append((quantity_add[feature], quantity_add[learning_feature]))
    G.add_edges_from(edges)

    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size=500, node_color="red")
    nx.draw_networkx_labels(G, pos, font_weight="bold")
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='blue', arrows=True)
    plt.title(wine + ' wine Network of Features')
    plt.show()

def task4(wine):
    data = pd.read_csv("winequality-" + wine + ".csv", sep=';')
    A = []
    B = []
    quality_target = 10
    i = 0
    for feature in data.columns[:-1]:
        X = data.drop(feature, axis=1)
        y = data[feature]
        regressor = LinearRegression()
        regressor.fit(X, y)
        coef = list(regressor.coef_)
        A.append(coef[0:i] + [-1] + coef[i:-1])
        B.append(-1 * regressor.intercept_ - 1 * quality_target * coef[-1])
        i += 1

    optimal = np.linalg.inv(np.array(A)).dot(np.array(B))
    optimal_rounded = [data.columns[i].upper() + ": " + str(round(optimal[i], 2)) for i in range(11)]
    print(optimal_rounded)
    X = data.drop('quality', axis=1)
    y = data['quality']
    regressor = LinearRegression()
    regressor.fit(X, y)
    score = regressor.score(X, y)
    print(score)

def task2b():
    data = pd.read_csv("winequality-white.csv", sep=';')
    X = data.drop('quality', axis=1)
    y = data['quality']

    regressor = LinearRegression()
    regressor.fit(X, y)
    white_coef1 = regressor.coef_
    print(regressor.coef_)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(X)
    white_coef2 = pca.explained_variance_


    data = pd.read_csv("winequality-red.csv", sep=';')
    X = data.drop('quality', axis=1)
    y = data['quality']

    regressor = LinearRegression()
    regressor.fit(X, y)
    red_coef1 = regressor.coef_
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(X)
    red_coef2 = pca.explained_variance_
    print(scipy.stats.pearsonr(white_coef1, red_coef1))
    print(scipy.stats.pearsonr(white_coef2, red_coef2))