import numpy as np
import matplotlib.pyplot as plt

class Markov:

    def __init__(self):
        self.points = []

    def transition_matrix(self, n):
        self.size = n
        self.array = np.eye(n, n)
        self.array2 = np.eye(n-1, n)
        
        state_0 = np.array(self.array[0])
        state_0.fill(0.2)
        state_0[n - 1] = 0.0

        state_i = np.array(self.array2)
        state_i[state_i == 1.0] = 0.8
        
        final_state = np.array(self.array[n - 1])
        final_state[-2] = 0.8

        self.array = np.eye(n, n)
        self.array[0] = state_0
        self.array[1 : n] = state_i
        self.array[n - 1] = final_state
        
        print(self.array)
    
    def propagate(self, steps):
        self.probability = np.zeros(self.size)
        self.probability[0] = 1.0 # initial vector (probability of 100%)

        init_probability = self.probability
        for i in range(0, steps + 1):
            self.probability = np.linalg.matrix_power(self.array, i) @ init_probability
            self.points.append(self.probability)
        
        print(self.probability)
    
    def plot(self, max_step):
        for i in range(0, max_step):
            plt.plot(np.arange(0, max_step), self.points[i], marker='o')
        
        plt.title("Markov Chain")
        plt.xlabel("states")
        plt.ylabel("probabibilities")
        plt.show()
        
