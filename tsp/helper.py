import json
from collections import defaultdict
import copy
import random


def isValid(inputList, preceDic):
    '''
    check if there exists any individual in inputList that does not satisfy the precedence constraint
    return a list of index of invalid instances
    '''

    result = []
    for idx, instance in enumerate(inputList):
        Flag = True

        for key in preceDic.keys():
            index_key = instance.index(key)

            for value in preceDic[key]:
                if instance.index(value) <= index_key:
                    Flag = False
                    break

            if Flag == False:
                break

        if Flag == False:
            result.append(idx)

    return result

def covertConditional(conditionalList):
    ### convert conditional constraint from list to dict

    ### decrease all index by 1 because array starts from index = 0 in python, while the original graph in the paper starts from index = 1
    for i in range(len(conditionalList)):
        conditionalList[i][0] -= 1
        conditionalList[i][1] -= 1

    ### from list to dict
    condiDic = defaultdict(list)
    for triple in conditionalList:
        condiDic[triple[0]].append([triple[1], triple[2]])

    return condiDic


def covertPrecedence(precedenceList):
    ### convert precedence constraint from list to dict

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


