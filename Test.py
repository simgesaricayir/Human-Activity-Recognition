import operator
import random
import joblib
import numpy
import pandas as pd
def getPredictions():
    test = pd.read_csv('UCI_HAR_dataset/csv_files/test.csv')
    #test = test.drop(['subject', 'Activity', 'ActivityName'], axis=1)

    models = ['lr.sav','dt.sav','svm.sav','rf.sav']
    pcas = ['lr_pca.sav', 'dt_pca.sav', 'svm_pca.sav', 'rf_pca.sav']
    for i,j in zip(models,pcas):
        print('model',i.split('.')[0])
        path = 'models/'
        path += i
        path2 =  'models/'
        path2+= j
        model = joblib.load(path)
        pca = joblib.load(path2)
        t = test.sample(1)
        print('real activity',t.ActivityName)
        t =t.drop(['subject', 'Activity', 'ActivityName'], axis=1)
        t = pca.transform(t)
        rslt = model.predict_proba(t)[0]
        prob_per_class_dictionary = dict(zip(model.classes_, rslt))
        prob_per_class_dictionary = dict(sorted(prob_per_class_dictionary.items(), key=operator.itemgetter(1), reverse=True))
        print(prob_per_class_dictionary,'\n')
        #print(list(prob_per_class_dictionary)[0])
getPredictions()