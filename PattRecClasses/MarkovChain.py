import numpy as np
from .DiscreteD import DiscreteD


class MarkovChain:
    """
    MarkovChain - class for first-order discrete Markov chain,
    representing discrete random sequence of integer "state" numbers.
    
    A Markov state sequence S(t), t=1..T
    is determined by fixed initial probabilities P[S(1)=j], and
    fixed transition probabilities P[S(t) | S(t-1)]
    
    A Markov chain with FINITE duration has a special END state,
    coded as nStates+1.
    The sequence generation stops at S(T), if S(T+1)=(nStates+1)
    """
    def __init__(self, initial_prob, transition_prob):

        self.q = initial_prob  #InitialProb(i)= P[S(1) = i]
        self.A = transition_prob #TransitionProb(i,j)= P[S(t)=j | S(t-1)=i]

        self.nStates = transition_prob.shape[1]

        self.is_finite = False
        if self.A.shape[0] != self.A.shape[1]:
            self.is_finite = True
            self.end_state= self.nStates-1


    def probDuration(self, tmax):
        """
        Probability mass of durations t=1...tMax, for a Markov Chain.
        Meaningful result only for finite-duration Markov Chain,
        as pD(:)== 0 for infinite-duration Markov Chain.
        
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.8.
        """
        pD = np.zeros(tmax)

        if self.is_finite:
            pSt = (np.eye(self.nStates)-self.A.T)@self.q    # eye 对角矩阵，@ matrix multiplication

            for t in range(tmax):
                pD[t] = np.sum(pSt)
                pSt = self.A.T@pSt

        return pD

    def probStateDuration(self, tmax):
        """
        Probability mass of state durations P[D=t], for t=1...tMax
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.7.
        """
        t = np.arange(tmax).reshape(1, -1)
        aii = np.diag(self.A).reshape(-1, 1)
        
        logpD = np.log(aii)*t + np.log(1-aii)
        pD = np.exp(logpD)

        return pD

    def meanStateDuration(self):
        """
        Expected value of number of time samples spent in each state
        """
        return 1/(1-np.diag(self.A))
    
    def rand(self, tmax):
        """
        S=rand(self, tmax) returns a random state sequence from given MarkovChain object.
        
        Input:
        tmax= scalar defining maximum length of desired state sequence.
           An infinite-duration MarkovChain always generates sequence of length=tmax
           A finite-duration MarkovChain may return shorter sequence,
           if END state was reached before tmax samples.
        
        Result:
        S= integer row vector with random state sequence, 整数行向量，state sequence generated
           NOT INCLUDING the END state,
           even if encountered within tmax samples
        If mc has INFINITE duration,
           length(S) == tmax
        If mc has FINITE duration,
           length(S) <= tmaxs
        """
        
        #*** Insert your own code here and remove the following error message

        S = np.empty(tmax, dtype=int)

        S[0] = np.random.choice(self.nStates, p=self.q)

        for i in range(1, tmax):
            prev_state = S[i - 1]
            pi = self.A[prev_state]
            S[i] = np.random.choice(self.nStates, p=pi)
            if self.is_finite and S[i] == self.end_state:
                return S[:i + 1]
        return S


    def viterbi(self):
        pass
    
    def stationaryProb(self):
        pass
    
    def stateEntropyRate(self):
        pass
    
    def setStationary(self):
        pass

    def logprob(self):
        pass

    def join(self):
        pass

    def initLeftRight(self):
        pass
    
    def initErgodic(self):
        pass

    def forward(self, pX):
        N = pX.shape[1]
        V = self.A.shape[0]
        alpha = np.zeros((V, N), dtype=np.float64)
        c = np.zeros(N + 1, dtype=np.float64)
        at = np.zeros(V, dtype=np.float64)

        at = self.q * pX[:, 0]
        c[0] = np.sum(at)
        alpha[:, 0] = np.divide(at, c[0])
        # print(alpha[:,0])

        for t in range(1, N):
            for j in range(V):
                at[j] = pX[j, t] * (alpha[:, t - 1] @ self.A[:, j])
            c[t] = np.sum(at)
            alpha[:, t] = at / c[t]
        if self.is_finite:
            c[N] = alpha[:, N - 1] @ self.A[:, self.A.shape[1] - 1]
        else:
            c = np.delete(c, N)

        return alpha, c

    def finiteDuration(self):
        pass

    def backward(self, c, pX):
        N = pX.shape[1]
        V = self.A.shape[0]
        beta = np.zeros((V, N), dtype=np.float64)

        if self.is_finite:
            beta[:, N - 1] = self.A[:, V] / (c[N] * c[N - 1])
        else:
            beta[:, N - 1] = 1 / c[N - 1]

        for i in range(N - 2, -1, -1):
            for j in range(V):
                beta[j, i] += self.A[j, 0:V] @ (pX[:, i + 1] * beta[:, i + 1])
                beta[j, i] = beta[j, i] / c[i]

        return beta


    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass
