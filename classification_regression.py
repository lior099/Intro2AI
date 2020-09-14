import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



import seaborn as sns; sns.set()


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, mean_squared_error, r2_score, explained_variance_score
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestRegressor

import scipy





def task1a(wine):
    data = pd.read_csv("winequality-"+wine+".csv", sep=';')
    X = data.drop('quality', axis=1)
    y = data['quality']

    y = label_binarize(y, classes=[quality for quality in range(11) if quality in y.values])
    n_classes = y.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True, random_state=0))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    print("done")

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        if not np.isnan(scipy.interp(all_fpr, fpr[i], tpr[i])).any():
            mean_tpr += scipy.interp(all_fpr, fpr[i], tpr[i])


    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='blue', linewidth=3)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='green', linewidth=3)


    plt.plot([0, 1], [0, 1], 'k--', lw=3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(wine+' wine RBF SVC')
    plt.legend(loc="lower right")
    plt.show()

    X = data.drop('quality', axis=1)
    y = data['quality']
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = y.values.reshape(-1, 1)
    y = sc_y.fit_transform(y)
    y = np.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    regressor = svm.SVR(kernel='linear')
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print(mean_squared_error(y_test, y_pred))
    print(r2_score(y_test, y_pred))
    print(explained_variance_score(y_test, y_pred))
    SVR_scores = [round(mean_squared_error(y_test, y_pred), 2),
                  round(r2_score(y_test, y_pred), 2),
                  round(explained_variance_score(y_test, y_pred), 2)]



    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print(mean_squared_error(y_test, y_pred))
    print(r2_score(y_test, y_pred))
    print(explained_variance_score(y_test, y_pred))
    RandomForest_scores = [round(mean_squared_error(y_test, y_pred), 2),
                           round(r2_score(y_test, y_pred), 2),
                           round(explained_variance_score(y_test, y_pred), 2)]
    cell_text = [SVR_scores, RandomForest_scores]
    row_labels = ["RBF SVR", "Random Forest"]
    col_labels = ["MSE", "R2 Score", "Explained Variance Score"]

    plt.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc="center right")
    plt.axis("off")
    plt.grid(False)
    plt.show()