import rpy2.robjects as robjects


def getTTestR(x_vet, y_vet):
    # USANDO o R para calcular t-test
    rd1 = (robjects.FloatVector(x_vet))
    rd2 = (robjects.FloatVector(y_vet))
    rvtest = robjects.r['t.test']
    pvalue = rvtest(rd1, rd2, paired=True)[2][0]

    return pvalue
