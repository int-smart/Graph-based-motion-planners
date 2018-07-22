import numpy as np
import copy
import openravepy
import math
import heapq
import time
#Can we add the states directly to the openQueue when we expand the parent as even if we get
# another with the same location the fvalue would be greater because now the edge cost
# would be more.

# Will edgeCost always be the euclidean distance not the manhattan.

# TODO Remove the collision points directly from the four and connected neighbor generation
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

class State(object):
    heuristic = ""
    connectionType = ""
    start = None
    goal = None
    stepSizeX = 0
    stepSizeY = 0
    stepSizeTheta = 0
    tolValue = 0

    def __init__(self, parent, state):
        self._parent = parent
        self._fValue = 0
        self._hValue = 0
        self._gValue = 0
        self.state = state

    @staticmethod
    def setParameters(startConfig, goalConfig, heuristic="manhattan", connectionType = "eight", stepSizeX = 0.0, stepSizeY = 0.0, stepSizeTheta = 0.0, tolValue = 0.0):
        State.heuristic = heuristic
        State.start = startConfig
        State.goal = goalConfig
        State.connectionType = connectionType
        State.stepSizeX = stepSizeX
        State.stepSizeY = stepSizeY
        State.stepSizeTheta = stepSizeTheta
        State.tolValue = tolValue

    @staticmethod
    def printParameters():
        print("The Astar parameters are : heuristic - ", State.heuristic, " connectionType - ", State.connectionType,
              " path - ", State.path, " start state - ", State.start, " goal - ", State.goal)

    def getTransitions(self):
        possibleStates = []
        if State.connectionType == "eight":
            possibleStates = getEightConnectedNeighbors(self.state,State.stepSizeX,State.stepSizeY,State.stepSizeTheta)
        else:
            possibleStates = getFourConnectedNeighbors(self.state,State.stepSizeX,State.stepSizeY,State.stepSizeTheta)
        return possibleStates

    def updategValues(self):
        if self._parent == None:
            self._gValue = 0
        else:
            # edgeCost = euclideanMetric(self._parent.state, self.state)
            if State.heuristic == "manhattan":
                edgeCost = manhattanMetric(self._parent.state, self.state)
            else:
                edgeCost = euclideanMetric(self._parent.state, self.state)
            self._gValue = edgeCost + self._parent._gValue

    def updatehValues(self):
        if State.heuristic == "manhattan":
            self._hValue = manhattanMetric(self.state, State.goal)
        else:
            self._hValue = euclideanMetric(self.state, State.goal)

    def updatefValue(self):
        self.updategValues()
        self.updatehValues()
        self._fValue = self._hValue+self._gValue

    def getfValue(self):
        self.updatefValue()
        return self._fValue

    def getgValue(self):
        return self._gValue

    def isGoal(self):
        tolerance = euclideanMetric(self.state, State.goal)
        if tolerance<State.tolValue:
            return True
        else:
            return False
        # return (self.state == State.goal).all == True

    def generatePath(self, handler, env):
        path = []
        tempState = self
        while tempState is not None:
            path.append(tempState.state)
            tempState = tempState._parent
        path.reverse()
        for index in range(len(path)):
            drawState(handler, path[index], env, np.array(((0, 0, 0))))
        # time.sleep(30)
        return path

#Do the colliion checking before assigning the state and generating the neighbors and
#whatever ou feel
def updateValuesOfNode(nodesList, currentState):
    for node in nodesList:
        if (np.array(node.state) == np.array(currentState.state)).all == True:
            if node.getfValue() > currentState.getfValue():
                node = currentState


def Astar(start, stepSizeX, stepSizeY, stepSizeTheta, goal, env, robot, heuristicType, connectionType, tolValue):
    State.setParameters(start, goal, heuristicType, connectionType, stepSizeX, stepSizeY, stepSizeTheta, tolValue)
    path = []
    handler = []
    closedSet = set()
    nodesList = []

    #Declare priority queue
    priorityQueue = q.PriorityQueue()
    startState = State(None, start)
    startState.getfValue()

    addEntryToQueue(priorityQueue, startState.getfValue, startState)

    # Add the start state to visited set
    # closedSet.add(tuple(startState.state))

    while not priorityQueue.empty():            # while Q not empty do:
        # Get the current state from the Queue on the basis of the priority
        currentState = getEntryFromQueue(priorityQueue)[1]
        drawState(handler, currentState.state, env, np.array(((0, 0, 1))))
        # If the current state to expand is the goal
        if currentState.isGoal():
            print("The cost to the goal for ",State.heuristic,"heuristic considering ",State.connectionType," is ",currentState.getgValue())
            print("The fvalue to the goal for ", State.heuristic, "heuristic considering ", State.connectionType, " is ",
                  currentState.getfValue())
            return currentState.generatePath(handler, env)

        if tuple(currentState.state) in closedSet:
            updateValuesOfNode(nodesList, currentState)
            continue
        closedSet.add(tuple(currentState.state))
        nodesList.append(currentState)

        # Get the next states
        possibleStates = currentState.getTransitions()


        for transition in possibleStates:
            robot.SetActiveDOFValues(transition)
            if not env.CheckCollision(robot) and not(tuple(transition) in closedSet):
                tempState = State(currentState, transition)
                addEntryToQueue(priorityQueue, tempState.getfValue(), tempState)
                # closedSet.add(tuple(transition))
            elif env.CheckCollision(robot):
                drawState(handler, transition, env, np.array(((1, 0, 0))))

    return None


if __name__=="__main__":
    # path = Astar([0,0,0], 0.1, [1,2,3])
    # handler = []
    goal = [2.6,-1.3,0.1]
    # drawState(handler, goal, env)