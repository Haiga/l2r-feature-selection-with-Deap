import rpy2.robjects as robjects


def getTTestR(x_vet, y_vet):
    # USANDO o R para calcular t-test
    rd1 = (robjects.FloatVector(x_vet))
    rd2 = (robjects.FloatVector(y_vet))
    rvtest = robjects.r['t.test']
    pvalue = rvtest(rd1, rd2, paired=True)[2][0]

    # print rvtest(rd1, rd2, paired=True)

    # Usando o próprio python para calcular t-test
    # line = str(rvtest(rd1, rd2, paired=True))
    # text= line.split("\n")
    # pvalue= text[-6]
    # pvalue = rvtest(rd1, rd2, paired=True)
    # print "[", pvalue, "]"
    # print np.mean(x_vet), np.mean(y_vet), pvalue
    # pvalue = rvtest(rd1, rd2, paired=True)

    return pvalue
