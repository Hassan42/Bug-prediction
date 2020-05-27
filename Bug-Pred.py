import re
import javalang as jl
import numpy as np
import pandas as pd
import os
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


inputPath = "/Users/hassanatwi/Desktop/InformationModeling/ima-pr2/defects4j/tmp/src/com/google/javascript/jscomp"
BuggyInputPath = "/Users/hassanatwi/Desktop/InformationModeling/ima-pr2/defects4j/framework/projects/Closure/modified_classes"

def getNumberOfStatments(method):
    counter = 0
    for root, s in method:
        if type(s).__base__ is jl.tree.Statement and type(s) is not jl.tree.BlockStatement:
            counter += 1
    return counter

def getCondandLoop(method):
    counter = 0
    for root, s in method.filter(jl.tree.IfStatement):
        counter += 1
    for root, s in method.filter(jl.tree.SwitchStatement):
        counter += 1
    for root, s in method.filter(jl.tree.WhileStatement):
        counter += 1
    for root, s in method.filter(jl.tree.DoStatement):
        counter += 1
    for root, s in method.filter(jl.tree.ForStatement):
        counter += 1
    return counter

def getReturn(method):
    counter = 0
    for root, ret in method.filter(jl.tree.ReturnStatement):
        counter += 1
    return counter

def getThrow(method):
    counter = 0
    for root, th in method:
        if type(th) is jl.tree.MethodDeclaration:
            if th.throws is not None:
                counter += len(th.throws)
    return counter

def getMax(l):
    if len(l) != 0:
        return max(l)
    else:
        return 0

def isBuggy(classN):
    if classN in s:
        return 1
    else:
        return 0


df = pd.DataFrame(
    columns=['ClassName', 'MTH', 'FLD', 'RFC', 'INT', 'SZ', 'CPX', 'EX', 'RET', 'WRD', 'BCM', 'NML', 'DCM'])

for path, dirs, files in os.walk(inputPath):
    for name in files:
        if name.endswith(".java"):
            finalPath = str(path) + "/" + str(name)
            sc = open(finalPath, 'r').read()
            sc.replace('\n', '')
            tree = jl.parse.parse(sc)
            for root, c in tree.filter(jl.tree.ClassDeclaration):
                fileName = name.replace(".java","")
                if(c.name == fileName ):
                    MTH, FLD, RFC, INT = 0, 0, 0, 0
                    SZ, CPX, EX, RET = 0, 0, 0, 0
                    WRD, BCM, NML, DCM = 0, 0, 0, 0
                    if c.implements != None:
                        INT = len(c.implements)
                    StatmentsList = []
                    CondList = []
                    Throwlist = []
                    ReturnList = []

                    totalNbOfLet = 0
                    for m in c.methods:
                        MTH += 1
                        if "public" in m.modifiers:
                            RFC += 1
                        RFC += len(list(m.filter(jl.tree.MethodInvocation)))
                        StatmentsList.append(getNumberOfStatments(m))
                        CondList.append(getCondandLoop(m))
                        Throwlist.append(getThrow(m))
                        ReturnList.append(getReturn(m))

                        totalNbOfLet += len(m.name)

                    if totalNbOfLet != 0:
                        NML = totalNbOfLet / MTH

                    SZ = getMax(StatmentsList)
                    CPX = getMax(CondList)
                    EX = getMax(Throwlist)
                    RET = getMax(ReturnList)
                    for v in c.fields:
                        FLD += 1
#                     for root3, am in c.filter(jl.tree.MethodInvocation):
#                         RFC += 1
                    for root4, d in c.filter(jl.tree.Documented):
                        if d.documentation != None:
                            BCM += 1
                            WRD += len(re.findall('\w+', d.documentation))
                    if sum(StatmentsList) != 0:
                        DCM = WRD / sum(StatmentsList)
                    df = df.append(
                        {'ClassName': c.name, 'MTH': MTH, 'FLD': FLD, 'RFC': RFC, 'INT': INT, 'SZ': SZ, 'CPX': CPX,
                         'EX': EX, 'RET': RET, 'WRD': WRD, 'BCM': BCM, 'NML': NML, 'DCM': DCM}, ignore_index=True)

s=""

for path, dirs, files in os.walk(BuggyInputPath):
    for name in files:
        if name.endswith(".src"):
            f = open(path + "/" + name, "r")
            s = s + f.read()

df["Buggy"] = df["ClassName"].apply(lambda x: isBuggy(x))


X = df.drop(['ClassName', 'Buggy'], axis=1)
y = df.Buggy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 80% training and 20% test

Clfs = []

clf = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=3)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
Clfs.append(clf)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

model = GaussianNB()
model = model.fit(X_train,y_train)
Clfs.append(model)
y_pred = model.predict(X_test)

clf = svm.LinearSVC(dual=False)
clf = clf.fit(X_train, y_train)
Clfs.append(clf)
y_pred = clf.predict(X_test)
precision_recall_fscore_support(y_test, y_pred, average='binary')

clf = MLPClassifier(hidden_layer_sizes=(12,12,12), max_iter = 400)
clf = clf.fit(X_train, y_train)
Clfs.append(clf)
y_pred = clf.predict(X_test)

clf = RandomForestClassifier(n_estimators = 100, criterion = "entropy", max_depth = 3)
clf = clf.fit(X_train, y_train)
Clfs.append(clf)
y_pred = clf.predict(X_test)

from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy="constant",constant = 1)
dummy_clf.fit(X_train, y_train)
Clfs.append(dummy_clf)
y_pred = dummy_clf.predict(X_test)
precision_recall_fscore_support(y_test, y_pred, average='binary')

from sklearn.model_selection import cross_validate

def getDfScores(clf):
    scores_df = pd.DataFrame()
    for i in range(20):
        scores = cross_validate(clf, X, y,
                        scoring={'f1': 'f1', 'precision': 'precision', 'recall': 'recall'})
        scores_df = scores_df.append(pd.DataFrame.from_dict(scores),ignore_index=True)
    return scores_df

fDf = pd.DataFrame()
pDf = pd.DataFrame()
rDf = pd.DataFrame()

for clf in Clfs:
    clf_name = str(clf).split("(")[0]
    mDf = getDfScores(clf)
    fDf[clf_name] = mDf.test_f1
    pDf[clf_name] = mDf.test_precision
    rDf[clf_name] = mDf.test_recall
    print("F1:" + clf_name + "  "+ str(mDf.test_f1.mean()) + "  "+ str(mDf.test_f1.std()))
    print("P:" + clf_name + "  "+ str(mDf.test_precision.mean()) + "  "+ str(mDf.test_precision.std()))
    print("R:" + clf_name + "  "+ str(mDf.test_recall.mean()) + "  "+ str(mDf.test_recall.std()))


fDf.name = "fDf"
pDf.name = "pDf"
rDf.name = "rDf"
dfs = [fDf,pDf,rDf]



from scipy.stats import wilcoxon
for df in dfs:
    cls = df.columns.tolist()
    for i in range(0,len(cls)-1):
        for j in range(i+1,len(cls)):
            (t, p) = wilcoxon(df[cls[i]], df[cls[j]])
            print("(" + str(cls[i]) + ", " +str(cls[j])+ ")")
    print(" ")