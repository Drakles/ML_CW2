# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from pacman import Directions
from game import Agent
import random
import game
import util


class Reward:
    GAME_OVER = -10
    FOOD = 1
    CAPSULE = 2
    DEFAULT = -0.04


# Class representing actions like going UP, RIGHT, DOWN or LEFT associated with tuple representing actual change on x,
# y axis.
class Actions:
    UP = (0, 1)
    RIGHT = (1, 0)
    DOWN = (0, -1)
    LEFT = (-1, 0)


# Dictionary to translate action to Direction class.
actionToDirection = {
    Actions.UP: Directions.NORTH,
    Actions.RIGHT: Directions.EAST,
    Actions.DOWN: Directions.SOUTH,
    Actions.LEFT: Directions.WEST,
}

# Dictionary to translate Direction class to action.
directionToAction = {
    Directions.NORTH: Actions.UP,
    Directions.EAST: Actions.RIGHT,
    Directions.SOUTH: Actions.DOWN,
    Directions.WEST: Actions.LEFT,
}


# this function is taken from api class of pacman project
def convert_to_food_list(state):
    # Returns a list of (x, y) pairs of food positions
    foodList = []
    foodGrid = state.data.food
    width = foodGrid.width
    height = foodGrid.height
    for i in range(width):
        for j in range(height):
            if foodGrid[i][j]:
                foodList.append((i, j))

    # Return list of food
    return foodList


class QState:

    def __init__(self, state):
        self.pacman_pos = state.getPacmanPosition()
        self.ghosts_pos = state.getGhostPositions()
        self.capsules = state.getCapsules()
        self.food = convert_to_food_list(state)

    def __eq__(self, otherGameStateData):
        return self.pacman_pos == otherGameStateData.pacman_pos and self.ghosts_pos == \
               otherGameStateData.ghosts_pos and self.capsules == otherGameStateData.capsules and self.food == \
               otherGameStateData.food


def pacman_next_pos(pacman_pos, action):
    return pacman_pos + directionToAction[action]


def getReward(pacman_position, food, capsules, ghosts_pos):
    if pacman_position in ghosts_pos:
        return Reward.GAME_OVER
    elif pacman_position in capsules:
        return Reward.CAPSULE
    elif pacman_position in food:
        return Reward.FOOD
    else:
        return Reward.DEFAULT


class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining=10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        # dictionary representing 2d table of states, actions and related Q value
        self.states_actions_q_val = {}

        self.prev_state = None
        self.prev_action = None

    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value

    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):

        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        print "Legal moves: ", legal
        print "Pacman position: ", state.getPacmanPosition()
        print "Ghost positions:", state.getGhostPositions()
        print "Food locations: "
        print state.getFood()
        print "Score: ", state.getScore()

        # Now pick what action to take. For now a random choice among
        # the legal moves
        action = random.choice(legal)
        qState = QState(state.gameStateData)

        reward = getReward(pacman_next_pos(qState.pacman_pos, action), qState.food, qState.capsules, qState.ghosts_pos)
        updateStatesActions(self.states_actions_q_val, reward, qState)

        # We have to return an action
        return action

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):

        print "A game just ended!"

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg, '-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)
