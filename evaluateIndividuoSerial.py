import l2rCodesSerial
import numpy as np


def getEval(individuo, NUM_GENES, X_train, y_train, X_test, y_test, query_id_test, ENSEMBLE, NTREES, SEED,
            DATASET, METRIC, NUM_FOLD, ALGORITHM):
    evaluation = []
    ndcg, queries = getPrecisionAndQueries(individuo, NUM_GENES, X_train, y_train, X_test, y_test, query_id_test,
                                           ENSEMBLE, NTREES, SEED, DATASET,
                                           METRIC)

    evaluation.append(ndcg)
    evaluation.append(getRisk(queries, DATASET, NUM_FOLD, ALGORITHM))
    evaluation.append(getTotalFeature(individuo))

    return evaluation


def getWeights(params):
    weights = []
    if 'precision' in params:
        weights.append(1)
    if 'risk' in params:
        weights.append(1)
    if 'feature' in params:
        weights.append(-1)

    return weights


def getPrecision(individuo, NUM_GENES, X_train, y_train, X_test, y_test, query_id_test, ENSEMBLE, NTREES, SEED, DATASET,
                 METRIC):
    ndcg, queries = getPrecisionAndQueries(individuo, NUM_GENES, X_train, y_train, X_test, y_test, query_id_test,
                                           ENSEMBLE, NTREES, SEED, DATASET,
                                           METRIC)
    return ndcg


def getTotalFeature(individuo):
    return sum([int(i) for i in individuo])


# PRECISA SER CORRIGIDA SE HOUVER MAIS DE UM BASEINE
def getRisk(queries, DATASET, NUM_FOLD, ALGORITHM):

    base = []

    arq = open(r'./baselines/' + DATASET + '/Fold' + NUM_FOLD + '/' + ALGORITHM + '.txt')
    for line in arq:
        base.append([float(line.split()[0])])
    basey = base.copy()

    for k in range(len(basey)):
        basey[k].append(queries[k])

    r = (l2rCodesSerial.getGeoRisk(np.array(basey), 0.1))[1]
    return r


def getPrecisionAndQueries(individuo, NUM_GENES, X_train, y_train, X_test, y_test, query_id_test, ENSEMBLE, NTREES,
                           SEED, DATASET,
                           METRIC):
    list_mask = list(individuo)
    features = []
    for i in range(NUM_GENES):
        if list_mask[i] == '1':
            features.append(i)
    X_train_ind = X_train[:, features]
    X_test_ind = X_test[:, features]
    #
    # X_train_ind = X_train.iloc[:, features]
    # X_test_ind = X_test.iloc[:, features]

    scoreTest = [0] * len(y_test)
    model = l2rCodesSerial.getTheModel(ENSEMBLE, NTREES, 0.3, SEED, DATASET)
    model.fit(X_train_ind, y_train)
    resScore = model.predict(X_test_ind)
    # resScore = model.predict(X_test_ind).to_array()
    c = 0
    for i in resScore:
        scoreTest[c] = i
        c = c + 1

    ndcg, queries = l2rCodesSerial.getEvaluation(scoreTest, query_id_test, y_test, DATASET, METRIC, "test")
    return ndcg, queries
