from markov import Markov

def main():
    Markov.transition_matrix(Markov, 10)

    Markov.propagate(Markov, 30)

if __name__ == "__main__":
    main()