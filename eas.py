import numpy as np
import matplotlib.pyplot as plt

# -------------------- Evolutionary Algorithm Class --------------------
class EvolutionaryAlgorithm():
    """
    Implements a simple evolutionary algorithm (EA) for optimizing a gene vector
    based on a fitness function.

    Parameters:
    - ff: Fitness function that evaluates a gene
    - gs: Gene size (number of parameters in the gene)
    - b: List of barrier/range values for each gene element [[min, max], ...]
    - mp: Mutation probability (controls mutation magnitude)
    - rp: Recombination probability (crossover rate)
    - g: Number of generations
    - p: Population size
    """
    def __init__(self, ff, gs, b, mp, rp, g, p):
        self.ff = ff  # Fitness function
        self.gs = gs  # Gene size
        self.b = b    # List of barriers for each gene element
        self.mp = mp  # Mutation probability
        self.rp = rp  # Recombination probability
        self.g = g    # Number of generations
        self.p = p    # Population size

        # Initialize population randomly within barriers
        self.i = np.zeros((self.p, self.gs))
        for i in range(self.p):
            for j in range(self.gs):
                # Random value scaled to gene-specific barriers
                self.i[i][j] = np.random.rand() * (self.b[j][1] - self.b[j][0]) + self.b[j][0]

        self.fHistory = np.zeros(self.g)  # Fitness history over generations
        self.t = self.g * self.p          # Total number of tournaments
        self.gene = self.i[0]             # Best gene so far

    # -------------------- Main EA Loop --------------------
    def run(self):
        for i in range(self.g):  # For each generation
            bestF = -1e9  # Track best fitness in generation

            for j in range(self.t):  # Tournament selection loop
                print(f"Running Gen {i + 1}/{self.g} Tournament {j + 1}/{self.t}")

                # Randomly select two individuals
                a = np.random.randint(0, self.p)
                b = np.random.randint(0, self.p)
                while a == b:
                    b = np.random.randint(0, self.p)

                # Evaluate their fitness
                fA = self.ff(self.i[a])
                fB = self.ff(self.i[b])

                # Determine winner and loser
                winner = a
                loser = b
                if fA < fB:
                    winner = b
                    loser = a

                # Update best fitness and gene
                winnerFitness = self.ff(self.i[winner])
                if winnerFitness >= bestF:
                    bestF = winnerFitness
                    self.gene = self.i[winner]

                # -------------------- Recombination --------------------
                for k in range(self.gs):
                    if np.random.random() < self.rp:
                        # Loser inherits gene element from winner
                        self.i[loser][k] = self.i[winner][k]

                # -------------------- Mutation --------------------
                for k in range(self.gs):
                    # Gaussian mutation based on mutation probability
                    mutation = np.random.normal(
                        self.mp * (-1 * (self.b[k][1] - self.b[k][0])) / 5,
                        self.mp * (self.b[k][1] - self.b[k][0]),
                        size=1
                    )
                    self.i[loser][k] += mutation
                    # Clip gene to allowed range
                    self.i[loser][k] = np.clip(self.i[loser][k], self.b[k][0], self.b[k][1])

            # Store best fitness of this generation
            self.fHistory[i] = bestF
            np.save(f"gene{i+1}.npy", self.gene)  # Optional: save gene per generation

        print(f"Fitness Achieved: {self.fHistory[-1]}")

        # -------------------- Plot Fitness History --------------------
        plt.plot(self.fHistory)
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title(f"Evolution with Population = {self.p}, Mutation Probability = {self.mp}, Recombination Probability = {self.rp}")
        plt.show()

        # Optional: save best overall gene for later use
        # np.save("gene.npy", self.gene)
