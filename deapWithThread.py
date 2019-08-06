import random
from deap import creator, base, tools, algorithms
import evaluateIndividuo
import l2rCodes
import json
import time
import numpy as np
import random
# import cudf
import threading
import matplotlib.pyplot as plt

NUM_INDIVIDUOS = 50
NUM_GENERATIONS = 2
NUM_GENES = None
PARAMS = ['precision', 'risk', 'feature']
#METHOD = 'spea2'  # 'nsga2'
DATASET = '2003_td_dataset'
NUM_FOLD = '1'

##
SEED = 14789
NTREES = 50
SUB_CROSS = 3
METRIC = 'NDCG'
ENSEMBLE = 4  # for regression
ALGORITHM = 'reg'  # for baseline
# ENSEMBLE = 1 #for regression forest
##

random.seed(SEED)

if DATASET == '2003_td_dataset':
    NUM_GENES = 64
elif DATASET == 'web10k':
    NUM_GENES = 136
elif DATASET == 'yahoo':
    NUM_GENES = 700
else:
    print('DATASET INVÁLIDO')



def main(DATASET, NUM_FOLD, NUM_GENES, METHOD):
    X_train, y_train, query_id_train = l2rCodes.load_L2R_file(
        './dataset/' + DATASET + '/Fold' + NUM_FOLD + '/Norm.' + 'train' + '.txt', '1' * NUM_GENES)
    X_test, y_test, query_id_test = l2rCodes.load_L2R_file(
        './dataset/' + DATASET + '/Fold' + NUM_FOLD + '/Norm.' + 'test' + '.txt', '1' * NUM_GENES)

    # X_train = cudf.DataFrame.from_records(X_train)
    # X_test = cudf.DataFrame.from_records(X_test)
    # y_train = cudf.Series(y_train)

    NOME_COLECAO_BASE = './resultados/' + DATASET + '-Fold' + NUM_FOLD + '-base-1.json'
    COLECAO_BASE = {}

    try:
        with open(NOME_COLECAO_BASE, 'r') as fp:
            COLECAO_BASE = json.load(fp)
        print('A base tem ' + str(len(COLECAO_BASE)) + ' indivíduos!\n')
    except:
        print('Primeira vez executando ...')

    def evalIndividuo(individual):
        evaluation = []
        individuo_ga = ''
        for i in range(NUM_GENES):
            individuo_ga += str(individual[i])
        if individuo_ga in COLECAO_BASE:
            if 'precision' in PARAMS:
                evaluation.append(COLECAO_BASE[individuo_ga]['precision'])
            if 'risk' in PARAMS:
                evaluation.append(COLECAO_BASE[individuo_ga]['risk'])
            if 'feature' in PARAMS:
                evaluation.append(COLECAO_BASE[individuo_ga]['feature'])
        else:
            result = evaluateIndividuo.getEval(individuo_ga, NUM_GENES, X_train, y_train, X_test, y_test, query_id_test,
                                               ENSEMBLE, NTREES, SEED, DATASET, METRIC, NUM_FOLD, ALGORITHM)
            COLECAO_BASE[individuo_ga] = {}
            COLECAO_BASE[individuo_ga]['precision'] = result[0]
            COLECAO_BASE[individuo_ga]['risk'] = result[1]
            COLECAO_BASE[individuo_ga]['feature'] = result[2]

            if 'precision' in PARAMS:
                evaluation.append(result[0])
            if 'risk' in PARAMS:
                evaluation.append(result[1])
            if 'feature' in PARAMS:
                evaluation.append(result[2])
        individual.fitness.values = evaluation
        return evaluation

        # v = 0
        # for i in range(len(individual)):
        #     if i > NUM_GENES / 2:
        #         v += individual[i]
        #     else:
        #         v -= individual[i]
        # return sum(individual), v,

    creator.create("MyFitness", base.Fitness, weights=evaluateIndividuo.getWeights(PARAMS))
    creator.create("Individual", list, fitness=creator.MyFitness)

    toolbox = base.Toolbox()

    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=NUM_GENES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evalIndividuo)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    if METHOD == 'spea2':
        toolbox.register("select", tools.selSPEA2)
    elif METHOD == 'nsga2':
        toolbox.register("select", tools.selNSGA2)
    else:
        Exception()

    temp1 = time.time()
    population = toolbox.population(n=NUM_INDIVIDUOS)
    if METHOD == 'nsga2':
        population = toolbox.select(population, NUM_INDIVIDUOS)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    # logbook.header = "gen", "evals", "std", "min", "avg", "max"
    logbook.header = "gen", "min", "max"

    for gen in range(NUM_GENERATIONS):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)

        if METHOD == 'nsga2':
            threads = []
            for ind in population + offspring:
                x = threading.Thread(target=evalIndividuo, args=(ind,))
                threads.append(x)
                x.start()
            for thread in threads:
                thread.join()
            # fits = toolbox.map(toolbox.evaluate, population + offspring)
            # for fit, ind in zip(fits, population + offspring):
            #     ind.fitness.values = fit

        elif METHOD == 'spea2':
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit

        population = toolbox.select(population + offspring, k=NUM_INDIVIDUOS)

        record = stats.compile(population)
        logbook.record(gen=gen, **record)
        #print(logbook.stream)

    temp2 = time.time()
    print('tempo totaal == ' + str(temp2 - temp1))
    # top10 = tools.selNSGA2(individuals=population, k=10)

    # for ind in top10:
    #     print(ind)
    #     print(evalIndividuo(ind))
    # print(top10)

    # start = time.time()
    # with open(NOME_COLECAO_BASE, 'w') as fp:
    #     json.dump(COLECAO_BASE, fp)
    # end = time.time()
    #
    # print('Tempo de salvamento do arquivo: ' + str(end - start) + 's')

    # Dá pra fazer a evolução deles com as informações do logboook
    # front = np.array([ind.fitness.values for ind in population])
    # optimal_front = np.array(front)
    # plt.scatter(optimal_front[:, 0], optimal_front[:, 1], c="r")
    # plt.scatter(front[:, 0], front[:, 1], c="b")
    # plt.axis("tight")
    # plt.show()


# for method in ['nsga2', 'spea2']:
for method in ['nsga2']:
    # for method in ['spea2']:
    for i in range(1):
        print('Method: ' + method + ' fold' + str(i + 1) + '\n')
        main(DATASET, str(i + 1), NUM_GENES, method)