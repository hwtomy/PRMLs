import numpy as np
from PattRecClasses.MarkovChain import MarkovChain
from PattRecClasses.GaussD import GaussD
from PattRecClasses.HMM import HMM

def training(q, A, g, x):

    mc = MarkovChain(q, A)
    HMM1 = HMM(mc, g)
    pX = HMM1.probp(x)
    pX = pX / np.max(pX, axis=0)  # normalized
    #print(pX)

    alpha_hat, c = mc.forward(pX)
    alpha_hat = alpha_hat.T

    beta_hat = mc.backward(c, pX)
    beta_hat = beta_hat.T

    #print('alphahat0:',len(alpha_hat[0]))
    gamma = np.empty((alpha_hat.shape[0], alpha_hat.shape[1]))
    # print(gamma.shape)
    # print(alpha.shape)
    # print(beta.shape)
    # print(alpha.shape)
    # print(beta.shape)
    for i in range(alpha_hat.shape[1]):
        for j in range(alpha_hat.shape[0]):
            gamma[j, i] = alpha_hat[j, i] * beta_hat[j, i] * c[j]
    # print(gamma)

    # update q
    q_update = gamma[0, :]


    # update A
    ep = np.zeros((len(alpha_hat),A.shape[0],A.shape[1]))
    alpha_hat = np.array(alpha_hat).T
    beta_hat = np.array(beta_hat).T
    num_state = A.shape[0]
    epy = np.zeros(ep[0].shape)

    for t in range(len(ep) - 1):
        for i in range(num_state):
            for j in range(num_state):
                ep[t, j, i] = alpha_hat[i, t] * A[i, j] * pX[j, t + 1] * beta_hat[j, t + 1]
        epy = epy+ep[t]

    An = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            An[i, j] = epy[i, j] / np.sum(epy[i, :])



    # update B
    gamma = np.empty((alpha_hat.shape[0], alpha_hat.shape[1]))

    for i in range(alpha_hat.shape[1]):
        for j in range(alpha_hat.shape[0]):
            gamma[j, i] = alpha_hat[j, i] * beta_hat[j, i] * c[i]
    # print(gamma)
    data = x.T
    # update q
    q_update = gamma[:, 0]
    gam = np.array(gamma)
    # print(gam.shape)
    mean1 = np.zeros(data.shape)
    mean = np.zeros((len(g), len(data[0])))
    for i in range(len(g)):
        for j in range(data.shape[0]):
            mean1[j, :] = gam[i, j] * data[j, :]
        mean[i, :] = np.sum(mean1, axis=0)
        mean[i, :] = mean[i, :] / np.sum(gam[i, :])

    varn1 = np.zeros((len(g), len(data[0]), len(data[0])))
    for i in range(len(g)):
        for j in range(data.shape[0]):
            data1 = np.expand_dims(data[j, :] - mean[i], axis=0)
            data1 = data1.T @ data1
            varn1[i, :, :] = gam[i, j] * data1 + varn1[i, :, :]
        varn1[i, :, :] = varn1[i, :, :] / np.sum(gam[i, :])

    return q_update, An, mean, varn1




def htesting(q, A, g, x):


    mc = MarkovChain(q, A)
    HMM1 = HMM(mc, g)
    pX = HMM1.probp(x)
    # print(pX)
    pX = pX / np.max(pX, axis=0)  # normalized

    alpha, c = mc.forward(pX)
    alpha = alpha.T
    beta = mc.backward(c, pX)
    beta = beta.T

    gamma = np.empty((alpha.shape[0], alpha.shape[1]))

    for i in range(alpha.shape[1]):
        for j in range(alpha.shape[0]):
            gamma[j, i] = alpha[j, i] * beta[j, i] * c[i]

    staten = np.argmax(gamma.T, axis=0)

    return staten



