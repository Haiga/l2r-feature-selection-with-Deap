import l2rCodesSerial
import numpy as np
from sklearn import model_selection


def getEval2(individuo, NUM_GENES, X_train, y_train, X_test, y_test, query_id_train, ENSEMBLE, NTREES, SEED,
            DATASET, METRIC, NUM_FOLD, ALGORITHM, PARAMS):
    evaluation = []
    ndcg, queries = getPrecisionAndQueries(individuo, NUM_GENES, X_train, y_train, X_test, y_test, query_id_train,
                                           ENSEMBLE, NTREES, SEED, DATASET,
                                           METRIC)
    if 'precision' in PARAMS:
        evaluation.append(queries)
    else:
        evaluation.append(0)
    # evaluation.append(ndcg)
    if 'risk' in PARAMS:
        evaluation.append(round(getRisk(queries, DATASET, NUM_FOLD, ALGORITHM), 5))
    else:
        evaluation.append(0)
    if 'feature' in PARAMS:
        evaluation.append(getTotalFeature(individuo))
    else:
        evaluation.append(0)
    if 'trisk' in PARAMS:
        evaluation.append(getTRisk(queries, DATASET, NUM_FOLD, ALGORITHM))
    else:
        evaluation.append(0)

    return evaluation

def getEval(individuo, NUM_GENES, X_train, y_train, X_test, y_test, query_id_train, ENSEMBLE, NTREES, SEED,
            DATASET, METRIC, NUM_FOLD, ALGORITHM, PARAMS):
    evaluation = []
    
    if not ('novelty' in PARAMS or 'diversity' in PARAMS):
        ndcg, queries = getPrecisionAndQueries(individuo, NUM_GENES, X_train, y_train, X_test, y_test, query_id_train, ENSEMBLE, NTREES, SEED, DATASET, METRIC)
    
    if 'precision' in PARAMS:
        evaluation.append(queries)
    else:
        evaluation.append(0)
    # evaluation.append(ndcg)
    if 'risk' in PARAMS:
        evaluation.append(round(getRisk(queries, DATASET, NUM_FOLD, ALGORITHM), 5))
    else:
        evaluation.append(0)
    if 'feature' in PARAMS:
        evaluation.append(getTotalFeature(individuo))
    else:
        evaluation.append(0)
    if 'trisk' in PARAMS:
        evaluation.append(getTRisk(queries, DATASET, NUM_FOLD, ALGORITHM))
    else:
        evaluation.append(0)
    
    if 'novelty' in PARAMS or 'diversity' in PARAMS:
            score, label, listU = getScores(individuo, NUM_GENES, X_train, y_train, X_test, y_test, query_id_train, ENSEMBLE, NTREES, SEED, DATASET, METRIC)
            
    if 'novelty' in PARAMS:
        evaluation.append(getNovelty(score, label, listU))
    else:
        evaluation.append(0)

    if 'diversity' in PARAMS:
        evaluation.append(getDiversity(score, label, listU))
    else:
        evaluation.append(0)

    return evaluation

def getWeights(params):
    weights = []
    if 'precision' in params:
        weights.append(1)
    if 'risk' in params:
        weights.append(1)
    if 'feature' in params:
        weights.append(-1)
    if 'trisk' in params:
        weights.append(-1)

    return weights


def getPrecision(individuo, NUM_GENES, X_train, y_train, X_test, y_test, query_id_train, ENSEMBLE, NTREES, SEED,
                 DATASET,
                 METRIC):
    ndcg, queries = getPrecisionAndQueries(individuo, NUM_GENES, X_train, y_train, X_test, y_test, query_id_train,
                                           ENSEMBLE, NTREES, SEED, DATASET,
                                           METRIC)
    return ndcg


def getTotalFeature(individuo):
    return sum([int(i) for i in individuo])


# PRECISA SER CORRIGIDA SE HOUVER MAIS DE UM BASEINE
def getRisk(queries, DATASET, NUM_FOLD, ALGORITHM):
    base = []

    arq = open(r'./baselines/' + DATASET + '/Fold' + NUM_FOLD + '/' + ALGORITHM + 'train.txt')
    for line in arq:
        base.append([float(line.split()[0])])
    basey = base.copy()

    for k in range(len(basey)):
        basey[k].append(queries[k])

    r = (l2rCodesSerial.getGeoRisk(np.array(basey), 5))[1]
    return r


def getTRisk(queries, DATASET, NUM_FOLD, ALGORITHM):
    base = []

    arq = open(r'./baselines/' + DATASET + '/Fold' + NUM_FOLD + '/' + ALGORITHM + 'train.txt')
    for line in arq:
        base.append(float(line.split()[0]))

    r, vetorRisk = (l2rCodesSerial.getTRisk(queries, base, 5))
    return vetorRisk


def getPrecisionAndQueries(individuo, NUM_GENES, X_train, y_train, X_test, y_test, query_id_train, ENSEMBLE, NTREES,
                           SEED, DATASET,
                           METRIC):
    list_mask = list(individuo)
    features = []
    for i in range(NUM_GENES):
        if list_mask[i] == '1':
            features.append(i)
    X_train_ind = X_train[:, features]
    # X_test_ind = X_test[:, features]

    # queriesList = l2rCodesSerial.getQueries(query_id_train)
    # scoreTrain = [0] * len(y_train)
    #
    # kf = model_selection.KFold(n_splits=3, shuffle=True, random_state=SEED)
    # XF_train_index = []
    # XF_test_index = []
    #
    # for qtrain_index, qtest_index in kf.split(queriesList):
    #
    #     del XF_train_index[:]
    #     del XF_test_index[:]
    #
    #     for qtr in qtrain_index:
    #         XF_train_index = XF_train_index + queriesList[qtr]
    #     for qts in qtest_index:
    #         XF_test_index = XF_test_index + queriesList[qts]
    #
    #     XF_train, XF_test = X_train_ind[XF_train_index], X_train_ind[XF_test_index]  ####
    #     yf_train, yf_test = y_train[XF_train_index], y_train[XF_test_index]
    #     # qf_train, qf_test = query_id_train[XF_train_index], query_id_train[XF_test_index]
    #
    #     # resScore = None
    #     model = l2rCodesSerial.getTheModel(ENSEMBLE, NTREES, 0.3, SEED, DATASET)
    #
    #     model.fit(XF_train, yf_train)
    #     resScore = model.predict(XF_test)
    #
    #     # sortRespRel=[rel for (score, rel) in sorted(zip(resScore,yf_test), key=lambda pair: pair[0], reverse=True)]
    #     c = 0
    #     for i in XF_test_index:
    #         scoreTrain[i] = resScore[c]
    #         c = c + 1
    # ndcg, queries = l2rCodesSerial.getEvaluation(scoreTrain, query_id_train, y_train, DATASET, METRIC, "test")

    scoreTrain = [0] * len(y_train)

    #model = linear_model.LinearRegression(n_jobs=-1)
    model = l2rCodesSerial.getTheModel(ENSEMBLE, NTREES, 0.3, SEED, DATASET)


    model.fit(X_train_ind, y_train)
    resScore = model.predict(X_train_ind)

    # sortRespRel=[rel for (score, rel) in sorted(zip(resScore,yf_test), key=lambda pair: pair[0], reverse=True)]
    c = 0
    for i in range(len(resScore)):
        scoreTrain[i] = resScore[c]
        c = c + 1

    ndcg, queries = l2rCodesSerial.getEvaluation(scoreTrain, query_id_train, y_train, DATASET, 'ndcg', "test")

    return ndcg, queries


def getScores(individuo, NUM_GENES, X_train, y_train, X_test, y_test, query_id_train, ENSEMBLE, NTREES,
                           SEED, DATASET,
                           METRIC):
            
    list_mask = list(individuo)
    features = []
    for i in range(NUM_GENES):
        if list_mask[i] == '1':
            features.append(i)
    X_train_ind = X_train[:, features]

    scoreTrain = [0] * len(y_train)

    #model = linear_model.LinearRegression(n_jobs=-1)
    model = l2rCodesSerial.getTheModel(ENSEMBLE, NTREES, 0.3, SEED, DATASET)


    model.fit(X_train_ind, y_train)
    resScore = model.predict(X_train_ind)

    # sortRespRel=[rel for (score, rel) in sorted(zip(resScore,yf_test), key=lambda pair: pair[0], reverse=True)]
    c = 0
    for i in range(len(resScore)):
        scoreTrain[i] = resScore[c]
        c = c + 1
            
    return scoreTrain, y_train, query_id_train


import numpy as np
import math


def getQueries(query_id_train):
    queryList = []
    queriesList = []
    # query_id_train.append(-1)
    idQ = -1
    cDoc = 0
    for i in query_id_train:
        if idQ != i:
            if (len(queryList) > 0):
                queriesList.append(queryList[:])
                del queryList[:]

        idQ = i
        queryList.append(cDoc)
        cDoc = cDoc + 1

    queriesList.append(queryList)
    return queriesList


def disc(k):
    return math.pow(0.85, k - 1)


def p(listUsersEvaluateI, lengthU):
    return len(listUsersEvaluateI.tolist()) / lengthU


def getNovelty(score, label, listU):
    listUsers = getQueries(listU)
    lengthU = len(listUsers)
    lengthScore = len(score)

    lineNum = np.array(range(0, len(label)), dtype=int)
    mat = (np.vstack((np.reshape(score, -1), np.reshape(lineNum, -1)))).T

    listEPC = []
    for user in listUsers:

        matUser = mat[user]
        R = matUser[np.argsort(-matUser[:, 0], kind="mergesort")]

        sumDiscDiscounted = 0
        for k in range(lengthScore):
            sumDiscDiscounted += disc(k) * (1 - p(R[k][1], lengthU))

        C = 0
        for k in range(lengthScore):
            C += disc(k)

        C = 1 / C
        listEPC.append(C * sumDiscDiscounted)

    return np.array(listEPC)


def disclk(l, k):
    if 1 > l - k:
        return disc(1)
    else:
        return disc(l - k)


def d(Ui, Uj):
    lenIntersect = len(Ui.tolist())
    produtoRaizLenConjuntos = math.sqrt(len(Ui)) * math.sqrt(len(Uj))
    return (1 - lenIntersect) / produtoRaizLenConjuntos


def getDiversity(score, label, listU):
    listUsers = getQueries(listU)
    lengthU = len(listUsers)
    lengthScore = len(score)

    lineNum = np.array(range(0, len(label)), dtype=int)
    mat = (np.vstack((np.reshape(score, -1), np.reshape(lineNum, -1)))).T

    listEILD = []
    for user in listUsers:

        matUser = mat[user]
        R = matUser[np.argsort(-matUser[:, 0], kind="mergesort")]

        sumDiscDiscounted = 0
        for i in range(lengthScore):
            for j in range(lengthScore):
                if i != j:
                    sumDiscDiscounted += disc(i) * disclk(i, j) * d(R[i][1], R[j][1])  ##

        C = 0
        for k in range(lengthScore):
            C += disc(k)

        C = 1 / C

        Clinha = 0
        for i in range(lengthScore):
            for j in range(lengthScore):
                if i != j:
                    Clinha += disclk(i, j)
        Clinha = C / Clinha
        listEILD.append(Clinha * sumDiscDiscounted)

    return np.array(listEILD)
