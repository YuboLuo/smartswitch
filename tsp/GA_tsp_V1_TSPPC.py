import numpy as np, random, json, operator, pandas as pd, matplotlib.pyplot as plt
from helper import covertPrecedence, generateChild, isValid

'''
V1: TSP with Precedence Constraint (TSPPC)
build based on this post: https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35
also refer to this github page: https://github.com/ezstoltz/genetic-algorithm/blob/master/genetic_algorithm_TSP.ipynb

The above post only implements genetic algorithm for basic TSP, 
we added extra constraints and develop our code based on the above implementation
'''

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


def Fitness(route, switchMat):

    pathDistance = 0
    for i in range(0, len(route)):

        if i == len(route) - 1:
            break

        src = route[i]
        dst = route[(i + 1) % len(route)]

        pathDistance += switchMat[src][dst]

    fitness = 1 / float(pathDistance)

    return fitness


## Create our initial population

def initialPopulation(popSize, switchMat):
    population = []
    N, _ = switchMat.shape

    for i in range(0, popSize):
        population.append(random.sample(range(N), N))
    return population

def initialPopulation_TSPPC(popSize, switchMat, preceDic):
    population = []
    N, _ = switchMat.shape

    for i in range(0, popSize):
        sample = generateChild(range(N), preceDic)
        population.append(sample)
    return population


## Create the genetic algorithm

def rankRoutes(population, switchMat):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i], switchMat)
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)



def selection(popRanked, eliteSize):
    # selection process uses both Elitism and Fitness proportionate selection
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    # use Elitism to select
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])

    # use Fitness proportionate selection to select the rest
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
        child[i] = parent1[i]  # copy segment to the same position

    ptr = 0
    for item in parent2:
        if item not in childP1:
            while ptr < len(parent1) and child[ptr] != None: # find the next None element
                ptr += 1

            child[ptr] = item  # only copy the remaining gene to positions that are None
            ptr += 1

    return child



def breed_TSPPC(parent, preceDic):

    ### please refer to this paper: https://link.springer.com/content/pdf/10.1007/s10845-009-0296-4.pdf
    ### we use the crossover proposed in the above paper

    N = len(parent)
    index = random.sample(range(N), 2)
    low, high = min(index), max(index)

    ### randomly decide which part to be changed
    ### point = 0 -> [0: low]
    ### point = 1 -> [low: high + 1]
    ### point = 2 -> [high + 1, N]
    point = random.sample([0,1,2],1)[0]

    if point == 0:
        return generateChild(parent[0: low], preceDic) + parent[low: N]
    elif point == 1:
        return parent[0: low] + generateChild(parent[low: high + 1], preceDic) + parent[high + 1: N]
    elif point == 2:
        return parent[0: high + 1] + generateChild(parent[high + 1: N], preceDic)


    # toChange = [parent[0: low], parent[low: high + 1], parent[high + 1: N]][point]



def breedPopulation_TSPPC(matingpool, eliteSize, preceDic):
    children = []
    length = len(matingpool) - eliteSize
    # pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        # child = breed(pool[i], pool[len(matingpool) - i - 1])
        child = breed_TSPPC(matingpool[i + eliteSize], preceDic)
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


def nextGeneration_TSPPC(currentGen, eliteSize, mutationRate, switchMat, preceDic):
    popRanked = rankRoutes(currentGen, switchMat)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation_TSPPC(matingpool, eliteSize, preceDic)
    # print("BeforeMutation: ", isValid(children, preceDic))
    # nextGeneration = mutatePopulation(children, mutationRate)
    nextGeneration = children
    return nextGeneration

## do GA without plotting the progress
def geneticAlgorithm(popSize, eliteSize, mutationRate, generations, switchMat, preceDic):
    pop = initialPopulation_TSPPC(popSize, switchMat, preceDic)
    print("Initial distance: " + str(1 / rankRoutes(pop, switchMat)[0][1]))

    for i in range(0, generations):
        pop = nextGeneration_TSPPC(pop, eliteSize, mutationRate, switchMat, preceDic)

        if i % 10 == 0:
            print(str(i) + 'th generation finished')

    print("Final distance: " + str(1 / rankRoutes(pop, switchMat)[0][1]))
    bestRouteIndex = rankRoutes(pop, switchMat)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

## Plot the progress
def geneticAlgorithmPlot(popSize, eliteSize, mutationRate, generations, switchMat, preceDic):
    pop = initialPopulation_TSPPC(popSize, switchMat, preceDic)
    progress = []
    progress.append(1 / rankRoutes(pop, switchMat)[0][1])

    for i in range(0, generations):
        if i % 10 == 0:
            print(str(i) + 'th generation finished')
        pop = nextGeneration_TSPPC(pop, eliteSize, mutationRate, switchMat, preceDic)
        # print("Plot: ",isValid(pop, preceDic))
        # ranked = rankRoutes(pop, switchMat)
        # progress.append([1 / ranked[0][1], pop[ranked[0][0]]])
        progress.append(1 / rankRoutes(pop, switchMat)[0][1])



    print("Final distance: " + str(1 / rankRoutes(pop, switchMat)[0][1]))
    bestRouteIndex = rankRoutes(pop, switchMat)[0][0]
    bestRoute = pop[bestRouteIndex]


    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()
    return [e - 1 for e in bestRoute]



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


### randomly generate a list of cities and generate a corresponding switch overhead matrix
# cityList = []
# N_city = 5  # number of cities
# for i in range(0, N_city):
#     cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))
# switchMat = createMat(cityList)



with open('json_data.json') as json_file:
    data = json.load(json_file)


graph = data['Example']

instance = 0

precedenceList = graph[instance]['PrecedenceConstraint']
preceDic = covertPrecedence(precedenceList)

switchMat = np.array(graph[instance]['Matrix'])
node_num = len(switchMat)




# geneticAlgorithmPlot(popSize=100, eliteSize=20, mutationRate=0.01, generations=500, switchMat=switchMat)
optimal = geneticAlgorithmPlot(popSize=100, eliteSize=20, mutationRate=0.01, generations=40, switchMat=switchMat, preceDic=preceDic)
print([e + 1 for e in optimal])








