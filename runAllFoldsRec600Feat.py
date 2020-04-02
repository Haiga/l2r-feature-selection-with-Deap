import deapForL2r
#started 15:52
# base = "mv600"
# base = "last600"
# base = "you600"
base = "bib600"
num_features = 613
PARAMS = ['novelty', 'diversity', 'precision']
deapForL2r.main(base, str(0), num_features, 'spea2', PARAMS)
print('xxx1.1')
deapForL2r.main(base, str(0), num_features, 'nsga2', PARAMS)




PARAMS = ['novelty', 'precision']
print('xxx2.1')
deapForL2r.main(base, str(0), num_features, 'spea2', PARAMS)
print('xxx2.5')
deapForL2r.main(base, str(0), num_features, 'nsga2', PARAMS)


PARAMS = ['diversity', 'precision']
print('xxx3.1')
deapForL2r.main(base, str(0), num_features, 'spea2', PARAMS)
print('xxx3.5')
deapForL2r.main(base, str(0), num_features, 'nsga2', PARAMS)


PARAMS = ['novelty', 'diversity']
print('xxx4.1')
deapForL2r.main(base, str(0), num_features, 'spea2', PARAMS)
print('xxx4.4')
deapForL2r.main(base, str(0), num_features, 'nsga2', PARAMS)

