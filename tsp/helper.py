import json
from collections import defaultdict,deque
import copy
import random
import pandas as pd


def isValidIndiviual(individual, preceDic):
    '''
    check if the individual satisfies the precedence constraint defined in preceDic
    :param individual: one individual
    :return: True or False
    '''

    Flag = True

    for key in preceDic.keys():
        index_key = individual.index(key)

        for value in preceDic[key]:
            if individual.index(value) <= index_key:
                Flag = False
                break

        if Flag == False:
            break

    return Flag




def isValid(inputList, preceDic):
    '''
    check if there exists any individual in inputList that does not satisfy the precedence constraint
    return a list of index of invalid instances
    :param inputList is a list of individuals
    '''

    result = []
    for idx, instance in enumerate(inputList):
        # Flag = True
        #
        # for key in preceDic.keys():
        #     index_key = instance.index(key)
        #
        #     for value in preceDic[key]:
        #         if instance.index(value) <= index_key:
        #             Flag = False
        #             break
        #
        #     if Flag == False:
        #         break

        if isValidIndiviual(instance, preceDic) == False:
            result.append(idx)

    return result

def covertConditional(conditionalList):
    '''
    convert conditional constraint from list to dict
    :param conditionalList: the original ConditionalConstraint read from json file, it is a List
    :return: a dictionary
    '''


    ### decrease all index by 1 because array starts from index = 0 in python, while the original graph in the paper starts from index = 1
    for i in range(len(conditionalList)):
        conditionalList[i][0] -= 1
        conditionalList[i][1] -= 1

    ### from list to dict
    condiDic = defaultdict(list)
    for triple in conditionalList:
        if len(triple) == 3:
            condiDic[triple[0]].append([triple[1], triple[2]])
        else:
            condiDic[triple[0]].append([triple[1], 1])

    return condiDic

def covertCondition(condiList):
    '''
    convert condition constraint: we assume that there are only one-step conditions and no multiple conditions
    we use a tuple (<node1>, <node2>, <prob>) to represent that node2 is conditioned on node2 with a probability of prob to be executed
    examples for only one-step conditions: (1,2,0.1), (1,4,0.3), (3,5,0.6)
    examples for multiple-step conditions: (1,2,0.1), (2,5,0.5) where 2 is conditioned on 1 and 5 is conditioned on 2
                    multiple-step cases need more complex coding implementation, here we just want to show the proof-of-concept
                    so we only implement the easier one-step condition only case

    :param condiList: a list of tuple (<node1>, <node2>, <prob>)
    :return: a dict, since we are only dealing with one-step condition case, the precedence node is actually not important here
                we only return the later node and its execution probability
    '''

    condiDic = {}
    for cond in condiList:
        condiDic[cond[1]] = cond[2]

    return condiDic






def covertPrecedence(precedenceList, Type = 0):
    '''
    convert precedence constraint from list to dict
    :param precedenceList: the original PrecedenceConstraint read from json file, it is a List
    :param Type: whether use data from the paper (index starts from 1) or TSPLIB(SOP) dataset (index starts from 0)
    :return: a dictionary
    '''

    if Type == 0: # use data from the paper
        ### decrease all index by 1 because array starts from index = 0 in python, while the original graph in the paper starts from index = 1
        for i in range(len(precedenceList)):
            precedenceList[i][0] -= 1
            precedenceList[i][1] -= 1

    ### from list to dict
    preceDic = defaultdict(set)
    for pair in precedenceList:
        preceDic[pair[0]].add(pair[1])


    '''
    the original precedence relation list only includes immediate precedence relations
    we have to expand this to multiple steps, 
    e.g. if 1->2, 2->3, 3->4 exists, 1->3, 1->4, 2->4 also exist
    '''
    ### first write a function which expands one level deeper at a time
    def expandPrecedence(preceDic):

        ### expand the precedence relation one level down
        preceDic2 = copy.deepcopy(preceDic)
        keys = set(preceDic.keys())
        for key in keys:
            for index in preceDic[key]:

                if index in keys:
                    preceDic2[key] = preceDic2[key].union(preceDic[index])

        return preceDic2

    ### expandPrecedence() only expands one level deeper at a time
    ### we use a while loop to repeat until no more potential expansion exists
    while expandPrecedence(preceDic) != preceDic:
        preceDic = expandPrecedence(preceDic)

    return preceDic



def generateChild(NodeList, preceDic):
    '''
    generate a child from NodeList using Topological Sort based on the precedence constraints defined in preceDic
    :param NodeList: a list of numbers, could be a fraction of a complete individual
    :param preceDic: precedence constraint graph
    :return: a child
    '''

    ### if there is only one element, just directly return
    if len(NodeList) <= 1:
        return NodeList

    NodeSet = set(NodeList)

    ### create a new precedence dictionary called adjacent that only contains nodes in NodeList
    adjacent = {i:{} for i in NodeList}
    for key in preceDic.keys():
        if key in NodeSet:
            adjacent[key] = {elem for elem in preceDic[key] if elem in NodeSet}


    ### count the indegree according to the newly created adjacent
    indegree = { i:0 for i in NodeList}
    for value in adjacent.values():
        for index in value:
            indegree[index] += 1

    queue = [ key for key in indegree.keys() if indegree[key] == 0 ]
    output = []
    while queue:

        random.shuffle(queue)
        elem = queue.pop()
        output.append(elem)

        for adj in adjacent[elem]:
            indegree[adj] -= 1
            if indegree[adj] == 0:
                queue.append(adj)

    return output


def findAllTrips(route, condiDic):

    '''
    For TSP with conditional constraint, to evaluate one individual, we have to find out all possible
    routes when some tasks are skipped and their corresponding probability
    :param route: an individual to be evaluated
    :param condiDic: conditional constraint graph
    :return: all possible routes and their probability
    '''

    allTrips = []
    route = deque(route)

    def removeDependentTask(subroute, task):
        '''
        use recursion to remove all dependent tasks of the input task that will be skipped
        this function is called only when a task is skipped
        '''

        ### first check if this task exists, it might have already been removed before
        ### because one task may depend on multiple tasks
        if task in subroute:
            subroute.remove(task)

        ### remove in a recursive way
        if task in condiDic:
            for arc in condiDic[task]:
                removeDependentTask(subroute, arc[0])



    def DFS(subroute, prob, trip):
        '''
        use recursion (depth-first-search) to iterate all possible routes
        '''

        ### function call by reference, create local copies to avoid changing the original argument
        route_local = copy.deepcopy(subroute)
        task = route_local.popleft()

        trip_local = copy.deepcopy(trip)
        trip_local.append(task)

        ### finished iterating this subroute
        ### the final cost should be added to the total distance
        if len(route_local) == 0:
            allTrips.append([trip_local, prob])
            return


        if task in condiDic:
            # if task has at least one outgoing arc
            n_arc = len(condiDic[task]) # one node may have multiple outgoing conditional arcs
            prob /= n_arc  # split the probability evenly among each arc

            for arc in condiDic[task]:

                DFS(route_local, prob * arc[1], trip_local) # when the dependent task is not skipped
                if arc[1] < 1: # conditional execution, a new branch is created where the dependent task is skipped

                    removeDependentTask(route_local, arc[0])
                    if len(route_local) == 0:
                        allTrips.append([trip_local, prob * (1 - arc[1])])
                        return

                    DFS(route_local, prob * (1 - arc[1]), trip_local)

        else: # task has no arc which means this task has no dependent tasks
            DFS(route_local, prob, trip_local)

    initial_prob = 1
    initial_trip = []
    DFS(route, initial_prob, initial_trip)

    return allTrips


def readSOP(filename):
    '''
    TSPLIB has datasets for various TSP variants, http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
    We use Sequential Ordering Problem (SOP) dataset which has precedence constraints
    :return:
    '''

    ### we skip the first 8 lines and the last line of the file
    data = pd.read_csv(filename, delim_whitespace=True, skiprows=[i for i in range(8)], header=None)  # skip first 8 lines
    data = data.values[:-1]  # skip the last line

    '''
    suppose the city number is n, the data is a n*n matrix
    if (i,j) == -1, it means that j must be visited before i
                    if j must be visited before i, we will never need the travel cost from city i to city j
                    so it just uses this (i,j) to mark the precedence constraint 
    if (i,j) != -1, its value means the travel cost from city i to city j
    '''

    precedence = []
    n = len(data)
    for i in range(n):
        for j in range(n):
            if type(data[i][j]) != float:
                data[i][j] = float(data[i][j])

            if data[i][j] == -1:
                precedence.append([j, i])
                data[i][j] = float('inf')  # use inf to mark the precedence constraint

    return precedence, data



# with open('json_data.json') as json_file:
#     data = json.load(json_file)
#
# graph = data['Example']
#
# precedenceList = graph[0]['Precedence']
# matrix = graph[0]['Matrix']
# node_num = len(matrix)
#
#
# preceDic = covertPrecedence(precedenceList)
# print(generateChild([0,1,2,3], preceDic))
#
# print(isValid([[0,1,2,4,5,3,6]], preceDic))


