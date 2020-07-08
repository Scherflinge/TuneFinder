from sklearn import tree
# from sklearn.impute import SimpleImputer
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
import os
import numpy as np
import json
import math
import joblib
import FileManagement
import argparse


def main():
    parser = argparse.ArgumentParser("Tune Finder Model Tester")
    parser.add_argument("ModelPath", type=str,
                        help="Path for loading or saving the model")
    parser.add_argument("-tp", "--TestPath", type=str,
                        help="Path of features to test", dest="TestPath", default=None)
    parser.add_argument("-tf", "--TrainFolder", type=str,
                        required=False, help="Path of folder to train model", default=None, dest="TrainFolder")
    # parser.add_argument("-fn", "--FileNames", type=str,
    #                     help="Path to file containing names for results from model", default=None, dest="FileNames")

    args = vars(parser.parse_args())
    modelPath = args["ModelPath"]
    testPath = args["TestPath"]
    trainFolder = args["TrainFolder"]
    # fileNames = args["FileNames"]

    if not (testPath or trainFolder):
        print("No valid parameters given")
        exit()

    if not (os.path.exists(modelPath) or trainFolder):
        print("Model not supplied")
        exit()
    elif trainFolder:
        data, classes, classNames = loadData(trainFolder)
        model = createModel(data, classes)
        saveModel(modelPath, model, classNames)

    if testPath:
        if not os.path.exists(testPath):
            print("Path for testing doesn't exist")
            exit()
        testMachine(modelPath, testPath)
    # print(args)


def testMachine(modelPath, testPath):
    clf, filenames = loadModel(modelPath)
    names = filenames
    files = FileManagement.listAllFilesPerDir(testPath)
    for x in files.keys():
        featfiles = files[x]
        for y in featfiles:
            dat = loadTest(y)
            # print(dat)
            if dat != []:
                pred = clf.predict(dat).tolist()
                # print(pred)
                # cl = pred.index(max(pred[0]))
                cl = pred[0]
                if filenames:
                    cl = names[cl]
                print("Predicted: {}\nActual: {}\n".format(
                    os.path.basename(cl), os.path.basename(y)))

    # for x in files.keys():
    #     featfiles = files[x]
        # results = []
        # feats = []
        # for y in featfiles:
        #     feats.extend(loadTest(y))
        # if len(feats) == 0:
        #     continue

        # if hasattr(clf, "predict_proba"):
        #     a = clf.predict_proba(feats)
        #     # print(a)
        #     try:
        #         predict = math.floor(sum(a.tolist())/len(a.tolist())+0.5)
        #     except:
        #         predict = a
        # elif hasattr(clf, "predict"):
        #     a = clf.predict(feats)
        #     if isinstance(a, np.ndarray):
        #         predict = sum(a).tolist().index(max(sum(a)))
        #     else:
        #         predict = a
        # else:
        #     return

        # if names and isinstance(predict, int):
        #     print(names[predict])
        # else:
        #     print(predict)
        # print(x)
        # print()


def testModelOnFile(featurePath, testPath):
    data, cleanClasses, classes = loadData(featurePath)

    clf = svm.SVC(probability=True).fit(data, cleanClasses)
    joblib.dump(clf, "model.m")
    print(clf.predict_proba([loadTest(testPath)]))
    for i, x in enumerate(classes):
        print("{}: {}".format(i, x))


def loadModel(path):
    if os.path.exists(path):
        model, classes = joblib.load(path)
        # classes = joblib.load(os.path.join(
        # os.path.dirname(path), "filenames.m"))
        return model, classes
    else:
        return None, None


def saveModel(path, data, ClassNames):
    if not os.path.exists(os.path.dirname(path)):
        if os.path.dirname(path) is not "":
            os.makedirs(os.path.dirname(path))
    joblib.dump([data, ClassNames], path)
    # joblib.dump(ClassNames, os.path.join(os.path.dirname(path)+"FileNames.m"))


def loadTest(path):
    with open(path) as f:
        return json.loads(f.read())


def testModel(data, classes, iters=100):
    testdata = []
    testclass = []
    for i in range(iters):
        a = math.floor(random.random()*len(data))
        testdata.append(data.pop(a))
        testclass.append(classes.pop(a))

    clf = svm.SVC(probability=True)
    # clf = tree.DecisionTreeClassifier()

    clf = clf.fit(data, classes)
    right = 0
    for d, c in zip(testdata, testclass):
        # print()
        predictedArray = clf.predict_proba([d]).tolist()[0]
        # print(predictedArray)
        # print(c)

        predictedClass = predictedArray.index(max(predictedArray))
        if predictedClass == c:
            right += 1
    right /= len(testdata)
    return right


def loadData(path):
    data = []
    classes = []
    cleanClasses = []
    count = 0
    print("Loading Data...")

    allfiles = FileManagement.listAllFilesPerDir(path)
    for di in allfiles:
        classes.append(di)
        fileList = allfiles[di]
        for fName in fileList:
            with open(fName) as f:
                fData = json.loads(f.read())

            data.extend(fData)
            for i in range(len(fData)):
                cleanClasses.append(count)

        count += 1
    print("Loaded.")
    return data, cleanClasses, classes


def createModel(data, classes):

    print("Fitting Data... (This may take a while)")
    # clf = svm.SVC().fit(data, classes)
    clf = tree.DecisionTreeClassifier().fit(data, classes)
    # clf = KNeighborsRegressor(n_neighbors=10).fit(data, classes)

    print("Finished fitting model")
    return clf


if __name__ == "__main__":
    # trainFolder = "features"
    # modelPath = "model.m"
    # data, classNumbers, classNames = loadData(trainFolder)
    # model = loadModel(data, classNumbers)
    # saveModel(modelPath, model, classNames)
    main()
    # loadData("features")
    # testMachine("model.m", "testFeatures")
