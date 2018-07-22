import openravepy
import heapq
import numpy as np
import copy
import sys
import Queue
from collections import defaultdict
import math
import time
import matplotlib.pyplot as plt
try:
    import Queue as q
except ImportError:
    import queue as q

def addEntryToQueue(container, priority, entry):
    newEntry = copy.deepcopy(entry)
    container.put((priority, newEntry))
    # container.put((entry, priority))

def getEntryFromQueue(container):
    return container.get()

def euclideanMetric(current, goal):
    temp = [current[i]-goal[i] for i in range(len(current))]
    temp = np.asarray(temp)
    heuristic = np.linalg.norm(temp,2)
    return heuristic

def manhattanMetric(current, goal):
    temp = [current[i] - goal[i] for i in range(len(current))]
    temp = np.asarray(temp)
    heuristic = np.linalg.norm(temp,1)
    return heuristic

def getFourConnectedNeighbors(state, stepSizeX, stepSizeY, stepSizeTheta):
    temp = []
    nextCells = []
    for i in range(len(state)):
        for j in [-1,1]:
            temp = copy.deepcopy(state)
            if i==0:
                temp[i] = temp[i] + j * stepSizeX
            elif i==1:
                temp[i] = temp[i] + j * stepSizeY
            else:
                if (temp[i] + j * stepSizeTheta) >= math.pi:
                    Z = temp[i]-2*math.pi
                elif (temp[i] + j * stepSizeTheta) <= -1*math.pi:
                    Z = temp[i] + 2*math.pi
                else:
                    Z = temp[i]
                temp[i] = Z + j * stepSizeTheta
            nextCells.append(temp)
    return nextCells

def drawState(handler, state, env, clr = np.array(((0, 0, 1)))):
    point = copy.deepcopy(state)
    point[2] = 0.1
    handler.append(env.plot3(points=point, pointsize=8.0, colors=clr))

def getEightConnectedNeighbors(state, stepSizeX, stepSizeY, stepSizeTheta):
    temp = []
    nextCells = []

    for i in [-1,0,1]:
        for j in [-1,0,1]:
            for k in [-1,0,1]:
                if i==j==k==0:
                    continue
                if (state[2] + k * stepSizeTheta)>=math.pi:
                    Z = state[2]-2*math.pi
                elif (state[2] + k * stepSizeTheta) <= -1*math.pi :
                    Z = state[2] + 2*math.pi
                else:
                    Z = state[2]
                temp = [state[0] + i * stepSizeX, state[1] + j * stepSizeY, Z + k * stepSizeTheta]
                nextCells.append(temp)
    return nextCells

def plotGraph(vars):
    temp = vars[0][0]
    t_vars = []
    G_vars = []
    for i in range(len(vars)):
        vars[i][0] = vars[i][0] - temp
        G_vars.append(vars[i][1])

    for i in range(1,len(vars)):
        t_vars.append(vars[i][0])
    t_vars.append(vars[len(vars) - 1][0] + 110)
    G_vars[0] = sys.maxint
    plt.step(t_vars, G_vars, label="ANA*")
    plt.axis([0,t_vars[-1],10.6,11.2])
    plt.step([95.049, t_vars[-1]],[sys.maxint, 10.7605], label = "A*")
    plt.legend()
    plt.show()


class State(object):
    heuristic = ""
    connectionType = ""
    start = None
    goal = None
    stepSizeX = 0
    stepSizeY = 0
    stepSizeTheta = 0
    tolValue = 0
    G = float("inf")
    E = float("inf")
    env = None
    robot = None
    nodesDict = dict()

    def __init__(self, parent, state):
        self._parent = parent
        self._eValue = 0
        self._hValue = 0
        self._gValue = float("inf")
        self.state = state
        self.updatehValues()
        self.updateEValue()

    @staticmethod
    def setParameters(startConfig, goalConfig, env, robot, heuristic, connectionType, stepSizeX, stepSizeY, stepSizeTheta, tolValue):
        State.heuristic = heuristic
        State.start = startConfig
        State.goal = goalConfig
        State.connectionType = connectionType
        State.stepSizeX = stepSizeX
        State.stepSizeY = stepSizeY
        State.stepSizeTheta = stepSizeTheta
        State.tolValue = tolValue
        State.G = float("inf")
        State.E = float("inf")
        State.env = env
        State.robot = robot
        State.nodesDict = dict()
        State.handler = []
        State.plotVariables = []

    @staticmethod
    def printParameters():
        print("The Astar parameters are : heuristic - ", State.heuristic, " connectionType - ", State.connectionType,
              " path - ", State.path, " start state - ", State.start, " goal - ", State.goal)

    def setGValue(self, gValue):
        self._gValue = gValue

    def getGValue(self):
        return self._gValue

    def setEValue(self, eValue):
        self._eValue = eValue

    def updateEValue(self):
        self._eValue = self.calEValue()

    def getEValue(self):
        return self._eValue

    def getHValue(self):
        return self._hValue

    def calEValue(self):
        if State.G == float("inf"):
            return self.getHValue()
        else:
            return ((State.G - self.getGValue()) / self.getHValue())

    def edgeCost(self, transition):
        return euclideanMetric(self.state, transition.state)

    def getTransitions(self):
        allStates = []
        possibleStates = []
        if State.connectionType == "eight":
            allStates = getEightConnectedNeighbors(self.state,State.stepSizeX,State.stepSizeY,State.stepSizeTheta)
        else:
            allStates = getFourConnectedNeighbors(self.state,State.stepSizeX,State.stepSizeY,State.stepSizeTheta)
        for transition in allStates:
            State.robot.SetActiveDOFValues(transition)
            if not State.env.CheckCollision(State.robot):
                possibleStates.append(transition)
            else:
                drawState(State.handler, transition, State.env, np.array(((1, 0, 0))))
        return possibleStates

    def updatehValues(self):
        if State.heuristic == "manhattan":
            self._hValue = manhattanMetric(self.state, State.goal)
        else:
            self._hValue = euclideanMetric(self.state, State.goal)

    def isGoal(self):
        tolerance = euclideanMetric(self.state, State.goal)
        if tolerance<State.tolValue:
            return True
        else:
            return False
        # return (self.state == State.goal).all == True

    def generatePath(self):
        path = []
        tempState = self
        while tempState is not None:
            path.append(tempState.state)
            tempState = tempState._parent
        return path

    def setParent(self, parent):
        self._parent = parent

    def getParent(self):
        return self._parent

def updateValuesOfNode(nodesList, currentState):
    for node in nodesList:
        if (np.array(node.state) == np.array(currentState.state)).all == True:
            if node.getfValue() > currentState.getfValue():
                node = currentState

def updateTransition(State, openQueue):
    j = -1
    if State.G == float("inf"):
        j = 1
    for element in State.nodesDict.keys():
        idx = None
        for i in range(len(openQueue)):
            if tuple(openQueue[i][1].state) == element:
                idx = i
                break
        if idx is not None:
            openQueue[idx][1].setEValue(State.nodesDict[element].getEValue())
            openQueue[idx][1].setGValue(State.nodesDict[element].getGValue())
            openQueue[idx][1].setParent(State.nodesDict[element].getParent())
            if (openQueue[idx][1].getGValue()+openQueue[idx][1].getHValue()) >= State.G:
                openQueue.remove(openQueue[idx])
    for node in openQueue:
        node[1].updateEValue()
        node[0] = list(node[0])
        node[0][0] = j*node[1].getEValue()
        node[0] = tuple(node[0])

def drawState(handler, state, env, clr = np.array(((0, 0, 1)))):
    point = copy.deepcopy(state)
    point[2] = 0.1
    handler.append(env.plot3(points=point, pointsize=10.0, colors=clr))


# The insertion of elements is with negative values so that we can have the maximal elements out of the Queue.
def updateeValueTransition(temp, openQueue):
    tempState = copy.deepcopy(temp)
    j = -1
    if State.G==float("inf"):
        j = 1
    found = False
    for element in openQueue:
        if tuple((element[1]).state) == tuple(tempState.state):
            found = True
            element[0] = list(element[0])
            element[0][0] = j*tempState.getEValue()
            element[0] = tuple(element[0])
            element[1] = tempState
            break
    if found==False:
        heapq.heappush(openQueue, [(j*tempState.getEValue(),tempState.getGValue()), tempState])

def ImproveSolution(State, openQueue):
    visited = []
    while openQueue:
        tempNew = heapq.heappop(openQueue)
        temp = copy.deepcopy(tempNew)
        currentState = temp[1]

#########Comment the following code to leave the visited code block that is for optimization
        if tuple(currentState.state) in visited:
            continue
        visited.append(tuple(currentState.state))
##############################################################################################
        if not tuple(currentState.state) in State.nodesDict.keys():
            State.nodesDict[tuple(currentState.state)] = currentState

        # Make the global E equal to the least value of e of states
        if currentState.getEValue() < State.E:
            State.E = currentState.getEValue()


        # If the current state is the goal then return the gValue for the next iterations
        if currentState.isGoal():
            print("The G value is ", State.G)
            State.G = currentState.getGValue()
            print("The G value is ", State.G)
            State.plotVariables.append([time.time(), State.G])
            return currentState

        #Get the next transitions
        successors = currentState.getTransitions()

        for transition in successors:
            if tuple(transition) in State.nodesDict.keys():             #C
                tempState = State.nodesDict[tuple(transition)]
            else:
                tempState = State(currentState, transition)             #here the g value is infinite and evalue is hvalue. Creating new node to put in dictionary
                State.nodesDict[tuple(transition)] = tempState          #Putting the node in dictionary

            #The next statement is working fine each of the three components currentState.getGValue(), currentState.edgeCost(tempState), tempState.getGValue()
            if currentState.getGValue() + currentState.edgeCost(tempState)<tempState.getGValue():           #here getGValue will be used as the GValue of the previous iteration is used to change the GValue in this condition
                tempState.setGValue(currentState.getGValue() + currentState.edgeCost(tempState))    #G value is correctly assigned for the tempState
                tempState.setParent(currentState)                                                   #
                tempState.updateEValue()                                                            #Correct
                State.nodesDict[tuple(transition)] = tempState                                      #Correct
                drawState(State.handler, tempState.state, State.env, np.array(((0,0,1))))
                if tempState.getGValue()+tempState.getHValue()<State.G:                             #Correct
                    updateeValueTransition(tempState, openQueue)                                    #Correct1 ,
                    heapq.heapify(openQueue)


def Anastar(startConfig, goalConfig, env, robot, heuristic, connectionType, stepSizeX, stepSizeY, stepSizeTheta, tolValue):
    # Initialize the priority Queue
    openQueue = []

    State.setParameters(startConfig, goalConfig, env, robot, heuristic, connectionType, stepSizeX, stepSizeY, stepSizeTheta, tolValue)
    State.plotVariables.append([time.time(), State.G])

    startState = State(None, startConfig)
    # Set the gValue of the start state to zero and updating the EValue
    startState.setGValue(0)
    startState.updateEValue()

    #Push the start state to openQueue
    start = copy.deepcopy(startState)
    heapq.heappush(openQueue, [(start.getEValue(), start.getGValue()), start])
    endPoint = None
    while openQueue:
        endPoint = ImproveSolution(State, openQueue)
        #       path = parent       Report current E-suboptimal solution
        updateTransition(State, openQueue)
        heapq.heapify(openQueue)
        # updateeValueTransition(transition, updatedeValue, openQueue, G)
        if endPoint is not None:
            path = endPoint.generatePath()
    # plotGraph(State.plotVariables)
    print("The optimal solution has a value of ", State.G)
    path.reverse()
    for point in path:
        drawState(State.handler, point, State.env, np.array(((0,0,0))))
    return path


def main():
    Anastar()

if __name__=="__main__":
    main()