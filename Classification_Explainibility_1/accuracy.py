
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics, svm


def calculate_accuracy(x_train, x_test, y_train, y_test, model=None):
    clf = MultinomialNB().fit(x_train, y_train)
    if model == "SVM":
        clf = svm.SVC(kernel='linear').fit(x_train, y_train)
    predicted = clf.predict(x_test)
    accuracy_score = metrics.accuracy_score(y_test, predicted)
    return accuracy_score, predicted
