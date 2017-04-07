import numpy as np

class sample(object):
    def __init__(self):
        # new : [n_couches, c1, c2, c3, learning_rate, reg_l1, reg_l2, moment, decay, nesterov, activation]
        self.values = np.array([[0, 1, 2, 3], #n_couches
                                range(10, 500, 10), range(10, 500, 10), range(10, 500, 10), #couches
                                [0.001, 0.002, 0.004, 0.008, 0.016, 0.03, 0.06, 0.012, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8], #learning rate
                                [0.000001,0.00001,0.0001,0.001,0.01,0.1], #reg_l1
                                [0.000001,0.00001,0.0001,0.001,0.01,0.1], #reg_l2
                                [0.001, 0.002, 0.004, 0.008, 0.016, 0.03, 0.06, 0.012, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8], #moment
                                [.0,0.001, 0.002, 0.004, 0.008, 0.016, 0.03, 0.06, 0.012, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8], #decay
                                [0,1], #nesterov
                                [0, 1, 2]])
        self.max = np.zeros(self.values.shape[0], dtype='int')
        for i in range(self.values.shape[0]):
            self.max[i] = len(self.values[i])
        self.c = []
        for i in range(self.values.shape[0]):
            self.c.append(np.random.randint(self.max[i]))

    def get_MNIST(self):
        res = []
        res.append(self.values[0][self.c[0]])
        n = []
        for i in range(self.c[0]):
            n.append(self.values[1][self.c[1+i]])
        res.append(n)
        for i in range(4, self.values.shape[0]):
            res.append(self.values[i][self.c[i]])
        return res

    def gaussian_samp(self):
        s = sample()
        rand = np.random.normal(np.zeros(self.values.shape[0]), 0.5)
        for p in range(rand.shape[0]):
            s.c[p]=min(max(int(self.c[p]+0.5+rand[p]),0),self.max[p]-1)
        return s

    def get_RSM(self):
        s=self.get_MNIST()
        # new : [n_couches, noeuds, learning_rate, reg_l1, reg_l2, moment, decay, nesterov, activation]
        # train : [n_couches, c1, c2, c3, learning_rate, reg_l1, reg_l2, moment, decay, nesterov, a1, a2, a3]
        assert (len(s) == 9)
        t = np.array([s[0] / 3., 0, 0, 0, s[2], s[3], s[4], s[5], s[6], s[7], 0, 0, 0])
        t[10 + s[8]] = 1
        for n in range(len(s[1])):
            t[1 + n] = s[1][n] / 500.
        return t

    def __str__(self):
        return str(self.c)