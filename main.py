from markov import Markov
import matplotlib.pyplot as plt
import numpy as np

def main():
    markov = Markov()
    
    markov.transition_matrix(10, True)
    markov.propagate(30)
    markov.plot(10)
    
    plt.title("Markov Chain: Probability of being on each state")
    plt.xlabel("States")
    plt.ylabel("Probabibilities")
    plt.legend()
    plt.savefig("./exports/qsn3.png")
    plt.show()

    n_values = range(10, 41)
    markov.num_steps(n_values)

    n_samples = 20
    for _ in range(n_samples):
        states = markov.sample(0, 10, 20)
        plt.plot(states)
        
    plt.xlabel("Number of Steps")
    plt.ylabel("State")
    plt.savefig("./exports/qsn5.png")
    plt.show()

    n_samples = 1000
    n_steps = 100
    n_tm = 25
    averages = markov.average(0, n_tm, n_steps, n_samples)
    
    plt.plot(range(n_steps + 1), averages) 
    plt.xlabel("Time Step")
    plt.ylabel("Average State")
    plt.savefig("./exports/qsn6.png")
    plt.show()

    n_samples = 1000
    n_steps = 100
    n_tm = 25
    state_0 = 1
    record = []
    probability_distributions = []
    for i in range(0, n_samples):
        state = (markov.sample(state_0, n_tm, n_steps))[-1]
        probability_distributions = markov.propagate2(n_steps + 1).tolist()
        record.append(state)
    
    for i in range(0, len(probability_distributions)):
        probability_distributions[i] = probability_distributions[i] * n_samples
    
    plt.xlabel("States")
    plt.ylabel("Number of occurrences")
    plt.hist(record, bins=range(0, n_tm, 1), color='orange', edgecolor='black')
    plt.plot(probability_distributions, marker='o')
    plt.savefig("./exports/qsn7.png")
    plt.show()


if __name__ == "__main__":
    main()