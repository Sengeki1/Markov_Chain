from markov import Markov

def main():
    markov = Markov()
    markov.transition_matrix(10, True)
    markov.propagate(30)
    markov.plot(10)

    n_values = range(10, 41)
    markov.num_steps(n_values)

if __name__ == "__main__":
    main()