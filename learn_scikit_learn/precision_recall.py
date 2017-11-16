# -----------------------------------------------------------------------------
# Working from examples in the scikit_learn docs
#
# Precision and Recall
#     http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
# -----------------------------------------------------------------------------

import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

data = datasets.load_iris()
X = data.data
y = data.target

r = np.random.RandomState(11)
rows, cols = X.shape
X = np.c_[X, r.randn(rows, 200*cols)]
X_train, X_test, y_train, y_test = train_test_split(X[y<2], y[y<2],
                                                    test_size=0.5,
                                                    random_state=r)
clf = svm.LinearSVC(random_state=r)
clf.fit(X_train, y_train)
y_score = clf.decision_function(X_test)
precision_avg = average_precision_score(y_test, y_score)

print('Avg Precision-Recall: {0:0.2f}'.format(precision_avg))

precision, recall, tmp = precision_recall_curve(y_test, y_score)
plt.step(recall, precision, where='post')
plt.title('Binary Classification Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])


# Binary classification example