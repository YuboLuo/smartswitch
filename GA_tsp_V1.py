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

class Fitness999:
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



# def createRoute(cityList):
#     route = random.sample(cityList, len(cityList))
#     return route



# def initialPopulation(popSize, cityList):
#     population = []
#
#     for i in range(0, popSize):
#         population.append(createRoute(cityList))
#     return population

def initialPopulation999(popSize, switchMat):
    population = []
    N, _ = switchMat.shape

    for i in range(0, popSize):
        population.append(random.sample(range(N), N))
    return population

## Create the genetic algorithm




# def rankRoutes(population):
#     fitnessResults = {}
#     for i in range(0, len(population)):
#         fitnessResults[i] = Fitness(population[i]).routeFitness()
#     return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)

def rankRoutes999(population, switchMat):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness999(population[i], switchMat).routeFitness()
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




# def nextGeneration(currentGen, eliteSize, mutationRate):
#     popRanked = rankRoutes(currentGen)
#     selectionResults = selection(popRanked, eliteSize)
#     matingpool = matingPool(currentGen, selectionResults)
#     children = breedPopulation(matingpool, eliteSize)
#     nextGeneration = mutatePopulation(children, mutationRate)
#     return nextGeneration

def nextGeneration999(currentGen, eliteSize, mutationRate, switchMat):
    popRanked = rankRoutes999(currentGen, switchMat)
    selectionResults = selection(popRanked, eliteSize)    # no change for the function of selection
    matingpool = matingPool(currentGen, selectionResults) # no change for the function of matingPool
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration



# def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
#     pop = initialPopulation(popSize, population)
#     print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
#
#     for i in range(0, generations):
#         pop = nextGeneration(pop, eliteSize, mutationRate)
#
#     print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
#     bestRouteIndex = rankRoutes(pop)[0][0]
#     bestRoute = pop[bestRouteIndex]
#     return bestRoute

def geneticAlgorithm999(popSize, eliteSize, mutationRate, generations, switchMat):
    pop = initialPopulation999(popSize, switchMat)
    print("Initial distance: " + str(1 / rankRoutes999(pop, switchMat)[0][1]))

    for i in range(0, generations):
        pop = nextGeneration999(pop, eliteSize, mutationRate, switchMat)

        if i % 10 == 0:
            print(str(i) + 'th generation finished')

    print("Final distance: " + str(1 / rankRoutes999(pop, switchMat)[0][1]))
    bestRouteIndex = rankRoutes999(pop, switchMat)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

## Running the genetic algorithm


cityList = []

for i in range(0, 25):
    cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))



# geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)




## Plot the progress

def geneticAlgorithmPlot999(popSize, eliteSize, mutationRate, generations, switchMat):
    pop = initialPopulation999(popSize, switchMat)
    progress = []
    progress.append(1 / rankRoutes999(pop, switchMat)[0][1])

    for i in range(0, generations):
        if i % 10 == 0:
            print(str(i) + 'th generation finished')
        pop = nextGeneration999(pop, eliteSize, mutationRate, switchMat)
        progress.append(1 / rankRoutes999(pop, switchMat)[0][1])

    print("Final distance: " + str(1 / rankRoutes999(pop, switchMat)[0][1]))

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

# geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)
# geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)

'''
For SmartSwitch, the given input is not city coordinates but a switch overhead matrix, we have to change the above code 
to make it accept overhead matrix instead of a city list with coordinates
'''

# generate an overhead matrix based on the city list
def createMat(cityList):
    N = len(cityList)
    switchMat = np.zeros((N, N))

    for i in range(N - 1):
        for j in range(i + 1, N):
            switchMat[i][j] = cityList[i].distance(cityList[j])
            switchMat[j][i] = switchMat[i][j]

    return switchMat


switchMat = createMat(cityList)

# geneticAlgorithm999(popSize=100, eliteSize=20, mutationRate=0.01, generations=50, switchMat=switchMat)
geneticAlgorithmPlot999(popSize=100, eliteSize=20, mutationRate=0.01, generations=500, switchMat=switchMat)







