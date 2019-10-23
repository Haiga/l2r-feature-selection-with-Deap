import l2rCodesSerial


def save(DATASET, NUM_FOLD, ENSEMBLE, ALGORITHM, sparse=False):
    NUM_GENES = None
    SEED = 1313
    NTREES = 300
    SUB_CROSS = 3
    METRIC = 'NDCG'
    if DATASET == '2003_td_dataset':
        NUM_GENES = 64
    elif DATASET == 'web10k':
        NUM_GENES = 136
    elif DATASET == 'yahoo':
        NUM_GENES = 700
    elif DATASET in ['movielens', 'lastfm', 'bibsonomy', 'youtube']:
        NUM_GENES = 13

    else:
        print('DATASET INVÁLIDO')

    X_train, y_train, query_id_train = l2rCodesSerial.load_L2R_file(
        './dataset/' + DATASET + '/Fold' + NUM_FOLD + '/Norm.' + 'train' + '.txt', '1' * NUM_GENES, sparse)
    # X_test, y_test, query_id_test = l2rCodesSerial.load_L2R_file(
    #     './dataset/' + DATASET + '/Fold' + NUM_FOLD + '/Norm.' + 'test' + '.txt', '1' * NUM_GENES, sparse)

    scoreTest = [0] * len(y_train)
    model = l2rCodesSerial.getTheModel(1, NTREES, 0.3, SEED, DATASET)
    model.fit(X_train, y_train)
    resScore = model.predict(X_train)
    c = 0
    for i in resScore:
        scoreTest[c] = i
        c = c + 1

    ndcg, queries = l2rCodesSerial.getEvaluation(scoreTest, query_id_train, y_train, DATASET, METRIC, "test")

    f = open('./baselines/' + DATASET + '/Fold' + NUM_FOLD + '/' + ALGORITHM + 'train.txt', "w+")
    for i in range(len(queries)):
        f.write(str(queries[i]) + '\n')
        # f.write(str(queries[i]))
    f.close()
