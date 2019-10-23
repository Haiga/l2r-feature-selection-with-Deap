import deapForL2r
import gerarBaselineSerial
PARAMS = ['precision', 'risk']
gerarBaselineSerial.save('2003_td_dataset', '1', 4, 'reg', sparse=False)
# %cd /content/tcc_l2r/baselines
# !git add .
# %cd /content/tcc_l2r
# !git commit -m 'update baselines de train'
# !git push
gerarBaselineSerial.save('web10k', '2', 4, 'reg', sparse=False)
gerarBaselineSerial.save('web10k', '3', 4, 'reg', sparse=False)
gerarBaselineSerial.save('web10k', '4', 4, 'reg', sparse=False)
gerarBaselineSerial.save('web10k', '5', 4, 'reg', sparse=False)
for i in range(5):
    deapForL2r.main('web10k', str(i + 1), 136, 'spea2', PARAMS)
    # deapForL2r.main('2003_td_dataset', str(i + 1), 64, 'spea2', PARAMS)

# PARAMS = ['precision']
# for i in range(1):
#     deapForL2r.main('web10k', str(i + 1), 136, 'spea2', PARAMS)
#     # deapForL2r.main('2003_td_dataset', str(i + 1), 64, 'spea2', PARAMS)
#
# PARAMS = ['risk']
# for i in range(1):
#     deapForL2r.main('web10k', str(i + 1), 136, 'spea2', PARAMS)
#     # deapForL2r.main('2003_td_dataset', str(i + 1), 64, 'spea2', PARAMS)