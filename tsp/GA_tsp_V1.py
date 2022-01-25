import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt

## Create necessary classes and functions


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

class Fitness:
    def __init__(self, route, switchMat):
        self.route = route
        self.distance = 0
        self.fitness = 0.0
        self.mat = switchMat

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):

                src = self.route[i]
                dst = self.route[(i + 1) % len(self.route)]

                pathDistance += self.mat[src][dst]

            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness


## Create our initial population

def initialPopulation(popSize, switchMat):
    population = []
    N, _ = switchMat.shape

    for i in range(0, popSize):
        population.append(random.sample(range(N), N))
    return population

## Create the genetic algorithm

def rankRoutes(population, switchMat):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i], switchMat).routeFitness()
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)



def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults




def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


def breed(parent1, parent2):
    child = [None] * len(parent1)
    childP1 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        child[i] = parent1[i]

    ptr = 0
    for item in parent2:
        if item not in childP1:
            while ptr < len(parent1) and child[ptr] != None:
                ptr += 1

            child[ptr] = item
            ptr += 1

    return child



def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children


def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual



def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


def nextGeneration(currentGen, eliteSize, mutationRate, switchMat):
    popRanked = rankRoutes(currentGen, switchMat)
    selectionResults = selection(popRanked, eliteSize)    # no change for the function of selection
    matingpool = matingPool(currentGen, selectionResults) # no change for the function of matingPool
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

## do GA without plotting the progress
def geneticAlgorithm(popSize, eliteSize, mutationRate, generations, switchMat):
    pop = initialPopulation(popSize, switchMat)
    print("Initial distance: " + str(1 / rankRoutes(pop, switchMat)[0][1]))

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate, switchMat)

        if i % 10 == 0:
            print(str(i) + 'th generation finished')

    print("Final distance: " + str(1 / rankRoutes(pop, switchMat)[0][1]))
    bestRouteIndex = rankRoutes(pop, switchMat)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

## Plot the progress
def geneticAlgorithmPlot(popSize, eliteSize, mutationRate, generations, switchMat):
    pop = initialPopulation(popSize, switchMat)
    progress = []
    progress.append(1 / rankRoutes(pop, switchMat)[0][1])

    for i in range(0, generations):
        if i % 10 == 0:
            print(str(i) + 'th generation finished')
        pop = nextGeneration(pop, eliteSize, mutationRate, switchMat)
        progress.append(1 / rankRoutes(pop, switchMat)[0][1])

    print("Final distance: " + str(1 / rankRoutes(pop, switchMat)[0][1]))

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


# generate an overhead matrix based on the list of cities
def createMat(cityList):
    N = len(cityList)
    switchMat = np.zeros((N, N))

    for i in range(N - 1):
        for j in range(i + 1, N):
            switchMat[i][j] = cityList[i].distance(cityList[j])
            switchMat[j][i] = switchMat[i][j]

    return switchMat

##################################################################################
##################################################################################
##################     workflow starts here                 ######################


# ### randomly generate a list of cities and generate a corresponding switch overhead matrix
# cityList = []
# N_city = 25  # number of cities
# for i in range(0, N_city):
#     cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))
#
# switchMat = createMat(cityList)

### load dataset
data = pd.read_csv('dataset/dantzig42_d.txt', delim_whitespace=True, header=None)
switchMat = data.values

geneticAlgorithmPlot(popSize=100, eliteSize=20, mutationRate=0.01, generations=500, switchMat=switchMat)







