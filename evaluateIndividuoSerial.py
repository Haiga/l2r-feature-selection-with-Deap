import l2rCodesSerial
import numpy as np
from sklearn import model_selection
import math


def getEval(individuo, NUM_GENES, X_train, y_train, X_test, y_test, query_id_train, ENSEMBLE, NTREES, SEED,
            DATASET, METRIC, NUM_FOLD, ALGORITHM, PARAMS):
    evaluation = []
    ndcg, queries, scoreTest = getPrecisionAndQueries(individuo, NUM_GENES, X_train, y_train, X_test, y_test,
                                                      query_id_train,
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
    if 'novelty' in PARAMS:
        evaluation.append(getNovelty(scoreTest, y_train, query_id_train))
    else:
        evaluation.append(0)
    if 'diversity' in PARAMS:
        evaluation.append(getDiversity(scoreTest, y_train, query_id_train))
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
    if 'novelty' in params:
        weights.append(1)
    if 'diversity' in params:
        weights.append(1)
    return weights


def getPrecision(individuo, NUM_GENES, X_train, y_train, X_test, y_test, query_id_train, ENSEMBLE, NTREES, SEED,
                 DATASET,
                 METRIC):
    ndcg, queries, scoreTrain = getPrecisionAndQueries(individuo, NUM_GENES, X_train, y_train, X_test, y_test,
                                                       query_id_train,
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
    # list_mask = list(individuo)
    # features = []
    # for i in range(NUM_GENES):
    #     if list_mask[i] == '1':
    #         features.append(i)
    # X_train_ind = X_train[:, features]
    # # X_test_ind = X_test[:, features]
    #
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
    # return ndcg, queries, scoreTrain

    list_mask = list(individuo)
    features = []
    for i in range(NUM_GENES):
        if list_mask[i] == '1':
            features.append(i)
    X_train_ind = X_train[:, features]

    scoreTrain = [0] * len(y_train)

    # model = linear_model.LinearRegression(n_jobs=-1)
    model = l2rCodesSerial.getTheModel(ENSEMBLE, NTREES, 0.3, SEED, DATASET)

    model.fit(X_train_ind, y_train)
    resScore = model.predict(X_train_ind)

    # sortRespRel=[rel for (score, rel) in sorted(zip(resScore,yf_test), key=lambda pair: pair[0], reverse=True)]
    c = 0
    for i in range(len(resScore)):
        scoreTrain[i] = resScore[c]
        c = c + 1

    ndcg, queries = l2rCodesSerial.getEvaluation(scoreTrain, query_id_train, y_train, DATASET, METRIC, "test")
    return ndcg, queries, scoreTrain
    # return scoreTrain, y_train, query_id_train


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
    return listUsersEvaluateI / lengthU


# def p2(i, u):
#     return (pow(2, g(u, i)) - 1)/(pow(2, ))

def getNovelty(score, label, listU):
    listUsers = getQueries(listU)
    lengthU = len(listUsers)
    lengthScore = len(score)

    lineNum = np.array(range(0, len(label)), dtype=int)
    mat = (np.vstack((np.reshape(score, -1), np.reshape(lineNum, -1)))).T

    listEPC = []
    cont = 10
    for user in listUsers:

        matUser = mat[user]
        R = matUser[np.argsort(-matUser[:, 0], kind="mergesort")]
        lengthScore2 = len(matUser)
        sumDiscDiscounted = 0
        for k in range(lengthScore2):
            # sumDiscDiscounted += disc(k) * (1 - p(R[k][0], lengthU))
            # sumDiscDiscounted += disc(k) *p()* (1 - p(score[k], lengthU))
            sumDiscDiscounted += disc(k) * (1 - p(score[k], lengthU)) * score[k]

        C = 0
        for k in range(lengthScore2):
            C += disc(k)

        C = 1 / C
        listEPC.append(C * sumDiscDiscounted)
        # cont -= 1
        # if cont <= 0:
        #     break
    print('novelty: ' + str(np.mean(listEPC)))
    # print(np.mean(listEPC))
    return np.array(listEPC)


def disclk(l, k):
    if 1 > l - k:
        return disc(1)
    else:
        return disc(l - k)


def d(Ui, Uj, dif=1):
    # print(Ui)
    # print(Uj)
    produtoRaizLenConjuntos = math.sqrt(abs(Ui)) * math.sqrt(abs(Uj))
    return dif / produtoRaizLenConjuntos


def getDiversity(score, label, listU):
    listUsers = getQueries(listU)
    # lengthU = len(listUsers)
    # lengthScore = len(score)

    lineNum = np.array(range(0, len(label)), dtype=int)
    mat = (np.vstack((np.reshape(score, -1), np.reshape(lineNum, -1)))).T

    listEILD = []

    cont = 10
    for user in listUsers:

        matUser = mat[user]
        R = matUser[np.argsort(-matUser[:, 0], kind="mergesort")]
        lengthScore2 = len(matUser)
        sumDiscDiscounted = 0
        # print('inside for 1')

        # t1.start()
        for i in range(lengthScore2):
            for j in range(lengthScore2):
                if i != j:
                    sumDiscDiscounted += disc(i) * disclk(i, j) * d(R[i][0], R[j][0], abs(len(listUsers[i]) - len(listUsers[j])))  ##


        C = 0
        for k in range(lengthScore2):
            C += disc(k)

        C = 1 / C

        Clinha = 0
        for i in range(lengthScore2):
            for j in range(lengthScore2):
                if i != j:
                    Clinha += disclk(i, j)
        if Clinha != 0:
            Clinha = C / Clinha
        else:
            Clinha = 0
        listEILD.append(Clinha * sumDiscDiscounted)
        cont -= 1
        if cont <= 0:
            break
    print('diversity: ' + str(np.mean(listEILD)))
    # print(np.mean(listEILD))
    return np.array(listEILD)


def getHashUserRec(scores, listU):
    hashUser = {}
    for i, score in enumerate(scores):
        if listU[i] not in hashUser:
            hashUser[listU[i]] = []
        hashUser[listU[i]].append(score)
    return hashUser


def getUsers(listU):
    users = []
    for user in listU:
        if user not in users:
            users.append(user)
    return users


# import time
def getNovelty2(scores, label, listU):
    listUsers = getUsers(listU)
    hashUserRec = getHashUserRec(scores, listU)
    numero_users = len(listUsers)
    listEPC = []
    # s = time.time()
    for user in listUsers:
        num_recommendations = len(hashUserRec[user])
        soma = 0
        ordered_rec = sorted(hashUserRec[user])
        # print(max(ordered_rec))
        # print(ordered_rec[0])
        for k in range(num_recommendations):
            # soma += disc(k + 1) * (1 - 1 / numero_users) * ordered_rec[k]

            soma += disc(k + 1) * (1 - 1 / numero_users) * (pow(2, ordered_rec[k] - 1) / pow(2, max(ordered_rec)))
        C = 0
        for k in range(num_recommendations):
            C += disc(k + 1)
        listEPC.append((1 / C) * soma)
    # print('novelty: %s' % (np.mean(listEPC)))
    # e = time.time()
    # print('time: %ss' % (e - s))
    return np.array(listEPC)
