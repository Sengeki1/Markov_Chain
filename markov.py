import numpy as np
import matplotlib.pyplot as plt
import random

class Markov:

    def __init__(self):
        self.points = []

    def transition_matrix(self, n, enable_print):
        self.size = n
        self.tm = np.eye(n, n)
        self.array = np.eye(n - 1, n)
        
        state_0 = np.array(self.tm[0])
        state_0.fill(0.2)
        state_0[n - 1] = 0.0

        state_i = np.array(self.array)
        state_i[state_i == 1.0] = 0.8
        
        final_state = np.array(self.tm[n - 1])
        final_state[-2] = 0.8

        self.tm = np.eye(n, n)
        self.tm[0] = state_0
        self.tm[1 : n] = state_i
        self.tm[n - 1] = final_state
        
        if (enable_print):
            print("Transition Matrix: ", self.tm)
    
    def propagate(self, steps):
        self.probability = np.zeros(self.size)
        self.probability[0] = 1.0 # initial vector (probability of 100%)

        init_probability = self.probability
        for i in range(0, steps + 1):
            self.probability = np.linalg.matrix_power(self.tm, i) @ init_probability
            self.points.append(self.probability)
        
        print("Probability Vector final state: ",self.probability)
    
    def plot(self, max_step):
        for i in range(0, max_step):
            plt.plot(np.arange(0, max_step), self.points[i], marker='o', label=f"step {i}")
        
        plt.title("Markov Chain: Probability of being on each state")
        plt.xlabel("States")
        plt.ylabel("Probabibilities")
        plt.legend()
        plt.savefig("./exports/qsn3.png")
        plt.show()

    def num_steps(self, n_values):
        probability = 0.0
        steps = []
        i = 0

        for k in n_values:
            self.transition_matrix(k, False)
            self.probability = np.zeros(self.size)
            self.probability[0] = 1.0

            while probability < 0.5:
                probability = (np.linalg.matrix_power(self.tm, i) @ self.probability)[-1]
                i += 1

            print("Number of steps for the final state to be atleast 50%: ", i, "for n: ", k)
            steps.append(i)
            probability = 0.0
            i = 0
            
        plt.title("Number of steps for the final state to be atleast 50% for N")
        plt.xlabel("N values")
        plt.ylabel("Number of steps")
        plt.plot(n_values, steps)
        plt.savefig("./exports/qsn4c.png")
        plt.semilogy(n_values, steps)
        plt.savefig("./exports/qsn4c_semilogy.png")
        plt.show()

    def sample(self, initial_state, n, steps, n_samples):
        self.transition_matrix(n, False)
        states = np.zeros(steps + 1)

        for i in range(n_samples):
            state = initial_state
            probability = self.tm[:, state] # column of probability for state 0
            for k in range(steps + 1):
                state = state + 1 if state + 1 < 10 else state # next state
                rand = random.choices([0, state], [probability[0], probability[state]])
                if (rand[0] == 0): state = 0 # reset state if state is 0
                probability = self.tm[:, state] # next state probability

                states[k] = state 
            print(states)
            plt.plot(states)
        
        plt.xlabel("Number of Steps")
        plt.ylabel("State")
        plt.savefig("./exports/qsn5.png")
        plt.show()
