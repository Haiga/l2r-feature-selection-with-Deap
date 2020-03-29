import json
from rpy2 import robjects
import evaluateIndividuoSerial
import l2rCodesSerial
import numpy as np


def compare(x_vet, y_vet, min_p_value=0.1):
    # USANDO o R para calcular t-test
    rd1 = (robjects.FloatVector(x_vet))
    rd2 = (robjects.FloatVector(y_vet))
    rvtest = robjects.r['t.test']
    pvalue = rvtest(rd1, rd2, paired=True)[2][0]

    return pvalue < min_p_value, pvalue


# DATASETS = ['lastfm', 'movielens', 'youtube']
DATASETS = ['mv600']
# NUM_FEATURES = 13
NUM_FEATURES = 613
params = ['diversityprecision', 'noveltydiversity', 'noveltydiversityprecision', 'noveltyprecision']
full = '1' * NUM_FEATURES
# 6 is for SVM
# 7 is for MLP
# 1 is for RF
ENSEMBLES = [1, 6, 7]
NTREES = 20
SEED = 1887
NUM_FOLD = '0'
METRIC = "NDCG"
sparse = True
ALGORITHM = 'rf'

for dataset in DATASETS:
    for param in params:
        NOME_COLECAO_BASE = './22-03/r1/' + dataset + '-Fold' + '0obj_' + param + '.json'
        COLECAO_BASE = {}

        try:
            with open(NOME_COLECAO_BASE, 'r') as fp:
                COLECAO_BASE = json.load(fp)

            for item in COLECAO_BASE:
                for att in COLECAO_BASE[item]:
                    try:
                        if len(COLECAO_BASE[item][att]) > 1:
                            COLECAO_BASE[item][att] = np.array(COLECAO_BASE[item][att])
                    except:
                        pass
        except:
            print(param + ' ' + dataset + 'error')
            break

        # nsga é 1, spea é 2, 3 ambos
        bestNSGA = 0
        bestSPEA = 0
        bestindNSGA = ''
        bestindSPEA = ''
        if 'precision' in param:
            for ind in COLECAO_BASE:
                if COLECAO_BASE[ind]['method'] == 1 or COLECAO_BASE[ind]['method'] == 3:
                    mean = np.mean(COLECAO_BASE[ind]['precision'])
                    if mean > bestNSGA:
                        bestNSGA = mean
                        bestindNSGA = ind
                if COLECAO_BASE[ind]['method'] == 2 or COLECAO_BASE[ind]['method'] == 3:
                    mean = np.mean(COLECAO_BASE[ind]['precision'])
                    if mean > bestSPEA:
                        bestSPEA = mean
                        bestindSPEA = ind
        else:
            # if 'diversity' in param:
            for ind in COLECAO_BASE:
                if COLECAO_BASE[ind]['method'] == 1 or COLECAO_BASE[ind]['method'] == 3:
                    mean = np.mean(COLECAO_BASE[ind]['diversity'])
                    if mean > bestNSGA:
                        bestNSGA = mean
                        bestindNSGA = ind
                if COLECAO_BASE[ind]['method'] == 2 or COLECAO_BASE[ind]['method'] == 3:
                    mean = np.mean(COLECAO_BASE[ind]['diversity'])
                    if mean > bestSPEA:
                        bestSPEA = mean
                        bestindSPEA = ind

        print(param + ' ' + dataset + 'bestIndSpea:' + bestindSPEA)
        print(param + ' ' + dataset + 'bestIndNSGA:' + bestindNSGA)

        print('reading and training NSGA bestind ensemble ' + str(ENSEMBLE))
        nX_train, ny_train, nquery_id_train = l2rCodesSerial.load_L2R_file(
            './dataset/' + dataset + '/' + NUM_FOLD + '.' + 'train', bestindNSGA, sparse)
        nX_test, ny_test, nquery_id_test = l2rCodesSerial.load_L2R_file(
            './dataset/' + dataset + '/' + NUM_FOLD + '.' + 'test', bestindNSGA, sparse)

        print('reading and training SPEA bestind ensemble ' + str(ENSEMBLE))
        sX_train, sy_train, squery_id_train = l2rCodesSerial.load_L2R_file(
            './dataset/' + dataset + '/' + NUM_FOLD + '.' + 'train', bestindSPEA, sparse)
        sX_test, sy_test, squery_id_test = l2rCodesSerial.load_L2R_file(
            './dataset/' + dataset + '/' + NUM_FOLD + '.' + 'test', bestindSPEA, sparse)

        print('reading and training FULL ensemble ' + str(ENSEMBLE))
        fX_train, fy_train, fquery_id_train = l2rCodesSerial.load_L2R_file(
            './dataset/' + dataset + '/' + NUM_FOLD + '.' + 'train', full, sparse)
        fX_test, fy_test, fquery_id_test = l2rCodesSerial.load_L2R_file(
            './dataset/' + dataset + '/' + NUM_FOLD + '.' + 'test', full, sparse)

        # model = RF
        diversitys = []
        noveltys = []
        ndcgs = []

        for ENSEMBLE in ENSEMBLES:

            #
            # NSGA BEST IND ###
            #

            model = l2rCodesSerial.getTheModel(ENSEMBLE, NTREES, 0.3, SEED, dataset)
            model.fit(nX_train, ny_train)
            resScore = model.predict(nX_test)

            scoreTest = [0] * len(ny_test)
            c = 0
            for i in resScore:
                scoreTest[c] = i
                c = c + 1

            ndcg, queries = l2rCodesSerial.getEvaluation(scoreTest, nquery_id_test, ny_test, dataset, METRIC, "test")
            ndcgs.append(queries)
            diversitys.append(evaluateIndividuoSerial.getDiversity(scoreTest, ny_test, nquery_id_test))
            noveltys.append(evaluateIndividuoSerial.getNovelty(scoreTest, ny_test, nquery_id_test))

            #
            # SPEA BEST IND ###
            #

            model = l2rCodesSerial.getTheModel(ENSEMBLE, NTREES, 0.3, SEED, dataset)
            model.fit(sX_train, sy_train)
            resScore = model.predict(sX_test)

            scoreTest = [0] * len(sy_test)
            c = 0
            for i in resScore:
                scoreTest[c] = i
                c = c + 1

            ndcg, queries = l2rCodesSerial.getEvaluation(scoreTest, squery_id_test, sy_test, dataset, METRIC, "test")
            ndcgs.append(queries)
            diversitys.append(evaluateIndividuoSerial.getDiversity(scoreTest, sy_test, squery_id_test))
            noveltys.append(evaluateIndividuoSerial.getNovelty(scoreTest, sy_test, squery_id_test))

            #
            # FULL FEATURES ###
            #

            model = l2rCodesSerial.getTheModel(ENSEMBLE, NTREES, 0.3, SEED, dataset)
            model.fit(fX_train, fy_train)
            resScore = model.predict(fX_test)

            scoreTest = [0] * len(fy_test)
            c = 0
            for i in resScore:
                scoreTest[c] = i
                c = c + 1

            ndcg, queries = l2rCodesSerial.getEvaluation(scoreTest, fquery_id_test, fy_test, dataset, METRIC, "test")
            ndcgs.append(queries)
            diversitys.append(evaluateIndividuoSerial.getDiversity(scoreTest, fy_test, fquery_id_test))
            noveltys.append(evaluateIndividuoSerial.getNovelty(scoreTest, fy_test, fquery_id_test))

        print('Start Comparison:')
        # num = 6
        num = 3 * len(ENSEMBLES)
        for i in range(num):
            for j in range(num):
                if j > i:
                    print(compare(ndcgs[i], ndcgs[j]))


        for i in range(num):
            for j in range(num):
                if j > i:
                    print(compare(diversitys[i], diversitys[j]))

        for i in range(num):
            for j in range(num):
                if j > i:
                    print(compare(noveltys[i], noveltys[j]))


# Será feita uma comparação na seguinte ordem:
# Primeiro se compara a diversidade
# NSGA - SPEA - ENSEMBLE 1
# NSGA - FULL - ENSEMBLE 1
# NSGA - NSGA - ENSEMBLE 2
# NSGA - SPEA - ENSEMBLE 2
# NSGA - FULL - ENSEMBLE 2
# NSGA - NSGA - ENSEMBLE n
# NSGA - SPEA - ENSEMBLE n
# NSGA - FULL - ENSEMBLE n
# SPEA - FULL - ENSEMBLE 1
# SPEA - NSGA - ENSEMBLE 2
# SPEA - SPEA - ENSEMBLE 2
# SPEA - FULL - ENSEMBLE 2
# SPEA - NSGA - ENSEMBLE n
# SPEA - SPEA - ENSEMBLE n
# SPEA - FULL - ENSEMBLE n
# Segundo se compara a novidade, seguindo esse mesma ordem em cima
# Terceiro se compara a efetividade, seguindo esse mesma ordem em cima
