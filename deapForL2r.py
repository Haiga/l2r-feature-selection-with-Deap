# %cd /content/tcc_l2r
import random
from deap import creator, base, tools, algorithms
import evaluateIndividuoSerial
import l2rCodesSerial
import json
import time
import numpy as np
# import cudf
import controlTime as ct
import matplotlib.pyplot as plt
import readSintetic

NUM_INDIVIDUOS = 50  # 50
NUM_GENERATIONS = 50  # 50
NUM_GENES = None
PARAMS = ['precision', 'risk', 'feature']
METHOD = 'spea2'  # 'nsga2'
DATASET = '2003_td_dataset'
NUM_FOLD = '1'
SINTETIC = True
sparse = False

##
SEED = 1313

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
elif DATASET in ['movielens', 'lastfm', 'bibsonomy', 'youtube']:
    NUM_GENES = 13
    sparse = True
else:
    print('DATASET INVÁLIDO')

readFilesTimer = ct.Timer(nome='Tempo Leitura Dataset')
convertToDataFrameTimer = ct.Timer(nome='Tempo Conversão Array to CUDF')
readResultTimer = ct.Timer(nome='Tempo Leitura Fitness de Indivíduos Salvos')
avaliarTimer = ct.Timer(nome='Tempo Avaliação Indivíduo')
toolboxTimer = ct.Timer(nome='Tempo Criação das Classes da Toolbox')
populacaoInicialTimer = ct.Timer(nome='Tempo Geração de Pop. Inicial')
crossMutTimer = ct.Timer(nome='Tempo Crossover e Mutação')
atribuicaoFitTimer = ct.Timer(nome='Tempo Manipulação de Fitness da Toolbox')
methodTimer = ct.Timer(nome='Tempo Para Seleção de Indivíduos de Acordo com o Método')
persistResultTimer = ct.Timer(nome='Tempo Persistência de Dados no meio da Execução')
estatisticaGerTimer = ct.Timer(nome='Tempo para Computar Estatísticas da Geração')
printsTimer = ct.Timer(nome='Tempo para Printar Resultados')
persistFinalResultTimer = ct.Timer(nome='Tempo Para Persistir Dados no Final da Execução')


def main(DATASET, NUM_FOLD, NUM_GENES, METHOD):
    readFilesTimer.start()
    X_train, y_train, query_id_train = l2rCodesSerial.load_L2R_file(
        './dataset/' + DATASET + '/Fold' + NUM_FOLD + '/Norm.' + 'train' + '.txt', '1' * NUM_GENES, sparse)
    X_test, y_test, query_id_test = l2rCodesSerial.load_L2R_file(
        './dataset/' + DATASET + '/Fold' + NUM_FOLD + '/Norm.' + 'test' + '.txt', '1' * NUM_GENES, sparse)
    readFilesTimer.stop()

    # convertToDataFrameTimer.start()
    # X_train = cudf.DataFrame.from_records(X_train)
    # X_test = cudf.DataFrame.from_records(X_test)
    # y_train = cudf.Series(y_train)
    # convertToDataFrameTimer.stop()

    readResultTimer.start()
    NOME_COLECAO_BASE = './resultados/' + DATASET + '-Fold' + NUM_FOLD + '-base.json'
    COLECAO_BASE = {}

    try:
        with open(NOME_COLECAO_BASE, 'r') as fp:
            COLECAO_BASE = json.load(fp)
        printsTimer.start()
        print('A base tem ' + str(len(COLECAO_BASE)) + ' indivíduos!\n')
        printsTimer.stop()
    except:
        printsTimer.start()
        print('Primeira vez executando ...')
        printsTimer.stop()

    readResultTimer.stop()

    current_generation_s = 1
    current_generation_n = 1

    def evalIndividuo(individual):
        avaliarTimer.start()
        evaluation = []
        individuo_ga = ''
        for i in range(NUM_GENES):
            individuo_ga += str(individual[i])
        if '1' not in individuo_ga:
            if 'precision' in PARAMS:
                evaluation.append(0)
            if 'risk' in PARAMS:
                evaluation.append(0)
            if 'feature' in PARAMS:
                evaluation.append(NUM_GENES)
        elif individuo_ga in COLECAO_BASE:
            if 'precision' in PARAMS:
                evaluation.append(COLECAO_BASE[individuo_ga]['precision'])
            if 'risk' in PARAMS:
                evaluation.append(COLECAO_BASE[individuo_ga]['risk'])
            if 'feature' in PARAMS:
                evaluation.append(COLECAO_BASE[individuo_ga]['feature'])
        else:
            result = evaluateIndividuoSerial.getEval(individuo_ga, NUM_GENES, X_train, y_train, X_test, y_test,
                                                     query_id_test,
                                                     ENSEMBLE, NTREES, SEED, DATASET, METRIC, NUM_FOLD, ALGORITHM)
            COLECAO_BASE[individuo_ga] = {}
            COLECAO_BASE[individuo_ga]['precision'] = result[0]
            COLECAO_BASE[individuo_ga]['risk'] = result[1]
            COLECAO_BASE[individuo_ga]['feature'] = result[2]
            COLECAO_BASE[individuo_ga]['geracao_s'] = current_generation_s
            COLECAO_BASE[individuo_ga]['geracao_n'] = current_generation_n

            if 'precision' in PARAMS:
                evaluation.append(result[0])
            if 'risk' in PARAMS:
                evaluation.append(result[1])
            if 'feature' in PARAMS:
                evaluation.append(result[2])

        avaliarTimer.stop()
        return evaluation

    toolboxTimer.start()
    creator.create("MyFitness", base.Fitness, weights=evaluateIndividuoSerial.getWeights(PARAMS))
    creator.create("Individual", list, fitness=creator.MyFitness)

    toolbox = base.Toolbox()

    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=NUM_GENES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evalIndividuo)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.3)
    toolbox.register("selectTournament", tools.selTournament, tournsize=2)
    if METHOD == 'spea2':
        toolbox.register("select", tools.selSPEA2)
    elif METHOD == 'nsga2':
        toolbox.register("select", tools.selNSGA2)
    else:
        Exception()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    # logbook.header = "gen", "evals", "std", "min", "avg", "max"
    logbook.header = "gen", "min", "max"

    toolboxTimer.stop()

    populacaoInicialTimer.start()
    population = toolbox.population(n=NUM_INDIVIDUOS)
    if SINTETIC:
        list_individuos = readSintetic.get(DATASET, NUM_FOLD, NUM_INDIVIDUOS)
        for indice_individuo in range(NUM_INDIVIDUOS):
            temp_ind = list_individuos[indice_individuo]
            for indice_gene in range(NUM_GENES):
                population[indice_individuo][indice_gene] = temp_ind[indice_gene]

    if METHOD == 'nsga2':
        population = toolbox.select(population, NUM_INDIVIDUOS)
    archive = []
    populacaoInicialTimer.stop()

    for gen in range(NUM_GENERATIONS):
        if METHOD == 'nsga2':
            crossMutTimer.start()
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.9, mutpb=0.2)
            crossMutTimer.stop()

        if METHOD == 'nsga2':
            fits = toolbox.map(toolbox.evaluate, population + offspring)
            atribuicaoFitTimer.start()
            for fit, ind in zip(fits, population + offspring):
                ind.fitness.values = fit
            atribuicaoFitTimer.stop()

        elif METHOD == 'spea2':
            fits = toolbox.map(toolbox.evaluate, population)
            fitsA = toolbox.map(toolbox.evaluate, archive)
            atribuicaoFitTimer.start()
            for fit, ind in zip(fits, population):
                ind.fitness.values = fit
            for fit, ind in zip(fitsA, archive):
                ind.fitness.values = fit
            atribuicaoFitTimer.stop()

        if METHOD == 'nsga2':
            methodTimer.start()
            population = toolbox.select(population + offspring, k=NUM_INDIVIDUOS)
            methodTimer.stop()
        elif METHOD == 'spea2':
            methodTimer.start()
            archive = toolbox.select(population + archive, k=NUM_INDIVIDUOS)
            methodTimer.stop()

            mating_pool = toolbox.selectTournament(archive, k=NUM_INDIVIDUOS)
            offspring_pool = map(toolbox.clone, mating_pool)

            crossMutTimer.start()
            offspring_pool = algorithms.varAnd(offspring_pool, toolbox, cxpb=0.9, mutpb=0.2)
            crossMutTimer.stop()

            population = offspring_pool

        persistResultTimer.start()
        if gen % 5 == 0:
            with open(NOME_COLECAO_BASE, 'w') as fp:
                json.dump(COLECAO_BASE, fp)
        persistResultTimer.stop()

        estatisticaGerTimer.start()
        # print(population)
        # return 0
        if METHOD == 'nsga2':
            record = stats.compile(population)
            current_generation_n += 1
        elif METHOD == 'spea2':
            record = stats.compile(archive)
            current_generation_s += 1
        logbook.record(gen=gen, **record)
        estatisticaGerTimer.stop()
        printsTimer.start()
        print(logbook.stream)
        printsTimer.stop()

    # top10 = tools.selNSGA2(individuals=population, k=10)

    # for ind in top10:
    #     print(ind)
    #     print(evalIndividuo(ind))
    # print(top10)

    persistFinalResultTimer.start()
    with open(NOME_COLECAO_BASE, 'w') as fp:
        json.dump(COLECAO_BASE, fp)
    persistFinalResultTimer.stop()

    # Dá pra fazer a evolução deles com as informações do logboook
    # front = np.array([ind.fitness.values for ind in population])
    # optimal_front = np.array(front)
    # plt.scatter(optimal_front[:, 0], optimal_front[:, 1], c="r")
    # plt.scatter(front[:, 0], front[:, 1], c="b")
    # plt.axis("tight")
    # plt.show()

    if METHOD == 'nsga2':
        return population
    elif METHOD == 'spea2':
        return archive
