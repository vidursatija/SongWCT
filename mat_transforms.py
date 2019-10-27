import numpy as np

def whitening(fc):
    # f = [C, H*W]
    print(fc.shape)
    mc = np.mean(fc, axis=-1) # [C]
    mc = np.reshape(mc, [-1, 1])
    fc -= mc

    covar = np.matmul(fc, fc.T) # [C, C]
    print(covar.shape)

    eigenvalues, Ec = np.linalg.eigh(covar) # ([C], [C, C])
    print(eigenvalues)
    eigenvalues = np.abs(eigenvalues+1e-14)

    eigenvalues = np.power(eigenvalues, -0.5)
    Dc = np.diag(eigenvalues) # [C, C]

    mid = np.matmul(Ec, np.matmul(Dc, Ec.T))

    return np.matmul(mid, fc)

def colouring(fs, fc_hat):
    # f = [C, H*W]
    ms = np.mean(fs, axis=-1) # [C]
    ms = np.reshape(ms, [-1, 1])
    fs -= ms

    covar = np.matmul(fs, fs.T) # [C, C]

    eigenvalues, Es = np.linalg.eigh(covar) # ([C], [C, C])
    # print(eigenvalues)
    eigenvalues = np.abs(eigenvalues)

    eigenvalues = np.power(eigenvalues, 0.5)
    Dc = np.diag(eigenvalues) # [C, C]

    mid = np.matmul(Es, np.matmul(Dc, Es.T))

    return np.matmul(mid, fc_hat) + ms