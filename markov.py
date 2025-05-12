import numpy as np

class Markov:

    def __init__(self):
        self.array = np.empty()
        self.probability
        self.size

    def transition_matrix(self, n):
        self.size = n
        self.array = np.eye(n, n)
        
        state_0 = np.array(self.array[0])
        state_0.fill(0.2)
        state_0[n - 1] = 0.0

        state_i = np.array(self.array[1 : n - 1])
        state_i[state_i == 1.0] = 0.8
        
        final_state = np.array(self.array[n - 1])

        self.array = np.eye(n, n)
        self.array[0] = state_0
        self.array[1 : n - 1] = state_i
        self.array[n - 1] = final_state
        
        print(self.array)
    
    def propagate(self, steps):
        self.probability = np.zeros(self.size)
        self.probability[0] = 1.0 # initial vector (probability of 100%)

        final_prob = np.linalg.matrix_power(self.array, steps) @ self.probability
        
        print(final_prob)
        
        
