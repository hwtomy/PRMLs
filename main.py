from PattRecClasses.GaussD import GaussD
from traing_testing import htesting, training
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from averagedata import avgdata, avgseq
import cupy as cp


if __name__ == "__main__":
    g1 = GaussD(means=[0, 0, 0], cov=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    g2 = GaussD(means=[0, 0, 0], cov=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    g3 = GaussD(means=[0, 0, 0], cov=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    g = [g1, g2, g3]
    q = np.array([1, 0, 0])
    A = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])

    df = pd.read_csv('train.csv')
    usefuldata = df.iloc[:, 2:5]
    train_data = usefuldata.values.T
    #train_data = avgdata(train_data)

    df1 = pd.read_csv('test.csv')
    usefuldata1 = df1.iloc[:, 2:5]
    tdata = usefuldata1.values.T
    #tdata = avgdata(tdata)

    epochs = 20
    for i in tqdm(range(epochs)):
        q, A, meann, covn = training(q, A, g, train_data)  # first update based on initialized parameters
        g1 = GaussD(means=meann[0], cov=covn[0])
        g2 = GaussD(means=meann[1], cov=covn[1])
        g3 = GaussD(means=meann[2], cov=covn[2])
        g = [g1, g2, g3]

    print("q = \n", q)
    print("A = \n", A)
    print("mean = \n", meann)
    print("cov = \n", covn)

    # test
    state_seq = htesting(q, A, g, tdata)
    #state_seq = avgseq(state_seq)
    plt.plot(state_seq)
    plt.xlabel("time point")
    plt.ylabel("State")
    plt.title("Classification result")
    plt.show()
    fig, ax = plt.subplots()
    i = tdata.shape[0]
    for j in range(i):
        plt.plot(tdata[j, :])
    ax.legend(["x", "y", "z"])
    plt.show()


