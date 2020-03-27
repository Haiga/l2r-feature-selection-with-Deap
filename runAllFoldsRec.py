import deapForL2r13 as deapForL2r
import gerarBaselineSerial


PARAMS = ['novelty', 'diversity', 'precision']
print('1')
deapForL2r.main('lastfm', str(0), 13, 'spea2', PARAMS)
print('xxx1.1')
deapForL2r.main('movielens', str(0), 13, 'spea2', PARAMS)
deapForL2r.main('youtube', str(0), 13, 'spea2', PARAMS)
# print('xxx1.3')
# deapForL2r.main('bibsonomy', str(0), 13, 'spea2', PARAMS)

deapForL2r.main('lastfm', str(0), 13, 'nsga2', PARAMS)
print('xxx1.5')
deapForL2r.main('movielens', str(0), 13, 'nsga2', PARAMS)
deapForL2r.main('youtube', str(0), 13, 'nsga2', PARAMS)
# print('xxx1.7')
# deapForL2r.main('bibsonomy', str(0), 13, 'nsga2', PARAMS)



PARAMS = ['novelty', 'precision']
print('xxx2.1')
deapForL2r.main('lastfm', str(0), 13, 'spea2', PARAMS)
deapForL2r.main('movielens', str(0), 13, 'spea2', PARAMS)
print('xxx2.3')
deapForL2r.main('youtube', str(0), 13, 'spea2', PARAMS)
# deapForL2r.main('bibsonomy', str(0), 13, 'spea2', PARAMS)

print('xxx2.5')
deapForL2r.main('lastfm', str(0), 13, 'nsga2', PARAMS)
deapForL2r.main('movielens', str(0), 13, 'nsga2', PARAMS)
deapForL2r.main('youtube', str(0), 13, 'nsga2', PARAMS)
print('xxx2.7')
# deapForL2r.main('bibsonomy', str(0), 13, 'nsga2', PARAMS)



PARAMS = ['diversity', 'precision']
print('xxx3.1')
deapForL2r.main('lastfm', str(0), 13, 'spea2', PARAMS)
deapForL2r.main('movielens', str(0), 13, 'spea2', PARAMS)
deapForL2r.main('youtube', str(0), 13, 'spea2', PARAMS)
# print('xxx3.3')
# deapForL2r.main('bibsonomy', str(0), 13, 'spea2', PARAMS)

deapForL2r.main('lastfm', str(0), 13, 'nsga2', PARAMS)
deapForL2r.main('movielens', str(0), 13, 'nsga2', PARAMS)
print('xxx3.5')
deapForL2r.main('youtube', str(0), 13, 'nsga2', PARAMS)
# deapForL2r.main('bibsonomy', str(0), 13, 'nsga2', PARAMS)
# deapForL2r.main('lastfm', str(0), 13, 'nsga2', PARAMS)




PARAMS = ['novelty', 'diversity']
print('xxx4.1')
deapForL2r.main('lastfm', str(0), 13, 'spea2', PARAMS)
deapForL2r.main('movielens', str(0), 13, 'spea2', PARAMS)
deapForL2r.main('youtube', str(0), 13, 'spea2', PARAMS)
# deapForL2r.main('bibsonomy', str(0), 13, 'spea2', PARAMS)
print('xxx4.4')
deapForL2r.main('lastfm', str(0), 13, 'nsga2', PARAMS)
deapForL2r.main('movielens', str(0), 13, 'nsga2', PARAMS)
deapForL2r.main('youtube', str(0), 13, 'nsga2', PARAMS)
# deapForL2r.main('bibsonomy', str(0), 13, 'nsga2', PARAMS)
