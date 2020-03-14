import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC

train = pd.read_csv('UCI_HAR_dataset/csv_files/train.csv')
test = pd.read_csv('UCI_HAR_dataset/csv_files/test.csv')
frame = test+train
print(frame)
print(train.shape, test.shape)
print(train.head(3))


# gereksiz columnları çıkar
X_train = train.drop(['subject', 'Activity', 'ActivityName'], axis=1)
y_train = train.ActivityName


# get X_test and y_test from test csv file
X_test = test.drop(['subject', 'Activity', 'ActivityName'], axis=1)
y_test = test.ActivityName


print('X_train and y_train : ({},{})'.format(X_train.shape, y_train.shape))
print('X_test  and y_test  : ({},{})'.format(X_test.shape, y_test.shape))


plt.rcParams["font.family"] = 'DejaVu Sans'

def plot_confusion_matrix(cm, classes,normalize=False, title='Confusion matrix',cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def perform_model(filename, model, X_train, y_train, X_test, y_test):

    results = dict()
    train_start_time = datetime.now()
    print('training')
    pca = PCA(n_components=150)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    model.fit(X_train, y_train)

    joblib.dump(model,filename)
    filename = filename.split('.')[0]
    filename+='_pca.sav'
    joblib.dump(pca,filename)
    print('Done')
    train_end_time = datetime.now()
    results['training_time'] = train_end_time - train_start_time
    print('training time- {}\n\n'.format(results['training_time']))

    print('Predicting test data')
    test_start_time = datetime.now()
    y_pred = model.predict(X_test)
    test_end_time = datetime.now()
    print('Done \n \n')
    results['testing_time'] = test_end_time - test_start_time
    print('testing time(HH:MM:SS:ms) - {}\n\n'.format(results['testing_time']))
    results['predicted'] = y_pred

    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    results['accuracy'] = accuracy

    print('      Accuracy      ')
    print('\n    {}\n\n'.format(accuracy))

    cm = metrics.confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm
    print('Confusion Matrix')
    print('\n {}'.format(cm))

    print('Classifiction Report')
    classification_report = metrics.classification_report(y_test, y_pred)
    results['classification_report'] = classification_report
    print(classification_report)
    results['model'] = model
    return results




log_reg = linear_model.LogisticRegression()
filename = 'models/lr.sav'
perform_model(filename,log_reg, X_train, y_train, X_test, y_test)


rbf_svm = SVC(kernel='rbf',probability=True)
filename = 'models/svm.sav'
rbf_svm_grid_results = perform_model(filename,rbf_svm, X_train, y_train, X_test, y_test)


dt = DecisionTreeClassifier()
filename = 'models/dt.sav'
dt_grid_results = perform_model(filename,dt, X_train, y_train, X_test, y_test)


rfc = RandomForestClassifier()
filename = 'models/rf.sav'
perform_model(filename,rfc, X_train, y_train, X_test, y_test)




