from markov import Markov

def main():
    markov = Markov()
    markov.transition_matrix(10)
    markov.propagate(30)
    markov.plot(10)

if __name__ == "__main__":
    main()