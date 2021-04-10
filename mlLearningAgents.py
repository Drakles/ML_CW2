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
from copy import deepcopy

from pacman import Directions
from game import Agent
import random
import game
import util

"""
This class represents different rewards that the agent can get
"""
class Reward:
    WIN = 10000
    GAME_OVER = -1000
    FOOD = 100
    DEFAULT = -1


"""
Class representing actions like going UP, RIGHT, DOWN or LEFT associated
with tuple representing actual change on x, y axis.
"""
class Actions:
    UP = (0, 1)
    RIGHT = (1, 0)
    DOWN = (0, -1)
    LEFT = (-1, 0)


"""
This function returns all legal actions

@param legal: list of legal directions
@return: list of legal actions that can be made
"""
def getAllLegalActions(legal):
    return [directionToAction[legal_move] for legal_move in legal]

"""
Dictionary to translate action to Direction class.
"""
actionToDirection = {
    Actions.UP: Directions.NORTH,
    Actions.RIGHT: Directions.EAST,
    Actions.DOWN: Directions.SOUTH,
    Actions.LEFT: Directions.WEST,
}

"""
Dictionary to translate Direction class to action.
"""
directionToAction = {
    Directions.NORTH: Actions.UP,
    Directions.EAST: Actions.RIGHT,
    Directions.SOUTH: Actions.DOWN,
    Directions.WEST: Actions.LEFT,
}

"""
This function converts the location of elements in grid to list

@param grid: grid of elements to be converted
@return: list with positions of element in grid
"""
def convert_grid_to_list(grid):
    converted_list = []

    for i in range(grid.width):
        for j in range(grid.height):
            if grid[i][j]:
                converted_list.append((i, j))
    return converted_list


class QState:

    def __init__(self, pacman_pos, ghosts_pos, food_pos, walls):
        self.pacman_pos = pacman_pos
        self.ghosts_pos = ghosts_pos
        self.food_pos = food_pos
        self.walls = walls

    def __eq__(self, otherGameStateData):
        return self.pacman_pos == otherGameStateData.pacman_pos \
               and self.ghosts_pos == otherGameStateData.ghosts_pos \
               and self.food_pos == otherGameStateData.food_pos

    def __hash__(self):
        return hash((self.pacman_pos, self.ghosts_pos, self.food_pos))

    def __str__(self):
        return 'pacman:' + str(self.pacman_pos) + ' , ghosts:' + str(
            self.ghosts_pos) + ' food:' + str(self.food_pos)


def pacman_next_pos(pacman_pos, action):
    return pacman_pos + directionToAction[action]


# Returns dict with all possible new locations and associated moves without
# considering the walls
def possible_moves(pos):
    return {
        (pos[0], pos[1] + 1): Actions.UP,
        (pos[0] + 1, pos[1]): Actions.RIGHT,
        (pos[0], pos[1] - 1): Actions.DOWN,
        (pos[0] - 1, pos[1]): Actions.LEFT
    }


def search_pos_within_limit(pacman_pos, single_obj_pos, limit, walls_pos,
                            is_obj_found):
    if limit > 0 and not is_obj_found:

        if pacman_pos == single_obj_pos:
            return True

        for next_pos, action in possible_moves(pacman_pos).items():
            if next_pos not in walls_pos:
                if next_pos == single_obj_pos:
                    return True
                else:
                    is_obj_found = is_obj_found or \
                                   search_pos_within_limit(
                                       next_pos,
                                       single_obj_pos,
                                       limit - 1,
                                       walls_pos,
                                       is_obj_found)

        return is_obj_found
    else:
        return is_obj_found


def objects_within_range(pacman_pos, obj_pos, walls_pos, limit):
    obj_within_limit_distance = filter(
        lambda single_obj:
        util.manhattanDistance(pacman_pos, single_obj) <= limit, obj_pos)

    if obj_within_limit_distance:
        for single_obj_pos in obj_within_limit_distance:
            is_obj_detected = search_pos_within_limit(pacman_pos,
                                                      single_obj_pos,
                                                      limit,
                                                      walls_pos,
                                                      False)
            if is_obj_detected:
                return True

    return False


def getReward(pacman_position, food, ghosts_pos, walls):
    if pacman_position in ghosts_pos:
        return Reward.GAME_OVER
    elif pacman_position in food:
        return Reward.FOOD
    elif objects_within_range(pacman_position, food, walls, 1):
        return Reward.FOOD / 100
    else:
        return Reward.DEFAULT


def getQValue(action, state, stats_acts_q_val):
    actions_q_val = stats_acts_q_val.get(state, 0)
    if actions_q_val is not 0:
        return actions_q_val.get(action, 0)
    else:
        return actions_q_val


def next_position(pacman_pos, action):
    return tuple(map(sum, zip(pacman_pos, action)))


def max_next_q_values(pacman_pos, ghosts_pos, food_pos, stats_acts_q_val,
                      legal, walls):
    next_q_val = []

    for action in getAllLegalActions(legal):
        next_q_state = QState(next_position(pacman_pos, action),
                              ghosts_pos, food_pos, walls)
        next_q_val.append(getQValue(action, next_q_state, stats_acts_q_val))

    return max(next_q_val) if next_q_val else 0


def best_next_action(pacman_pos, ghosts_pos, food_pos, walls, stats_acts_q_val,
                     legal):
    best_q_val = float('-inf')
    best_action = None

    for action in getAllLegalActions(legal):
        next_q_state = QState(next_position(pacman_pos, action),
                              ghosts_pos, food_pos, walls)
        next_q_val = getQValue(action, next_q_state, stats_acts_q_val)
        if next_q_val >= best_q_val:
            best_q_val = next_q_val
            best_action = action

    return best_action


def e_greedy_action(legal, pacman_pos, ghosts_pos, food_pos, walls, e,
                    stats_acts_q_val):
    if random.uniform(0, 1.0) > e:
        return best_next_action(pacman_pos, ghosts_pos, food_pos, walls,
                                stats_acts_q_val, legal)
    else:
        random_action = random.choice(legal)
        return directionToAction[random_action]


# -p QLearnAgent -x 2000 -n 2010 -l smallGrid
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining = 10):
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

        # dictionary representing 2d table of states, actions and related Q
        # value
        self.stats_acts_q_val = {}

        self.prev_action = None
        self.prev_q_state = None

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

        pacman_pos = state.getPacmanPosition()
        ghosts_pos = tuple(state.getGhostPositions())
        food_pos = tuple(convert_grid_to_list(state.getFood()))
        walls = convert_grid_to_list(state.getWalls())

        # update
        if self.prev_q_state is not None:
            self.updateStatesActionsQValue(self.prev_action,
                                           self.prev_q_state,
                                           legal)

        action = e_greedy_action(legal, pacman_pos, ghosts_pos, food_pos, walls,
                                 self.epsilon, self.stats_acts_q_val)

        self.prev_action = action
        self.prev_q_state = QState(next_position(pacman_pos, action),
                                   ghosts_pos,
                                   food_pos, walls)

        return actionToDirection[action]

    def updateStatesActionsQValue(self, action, q_state, legal):
        if q_state not in self.stats_acts_q_val:
            self.stats_acts_q_val[q_state] = {}

        q_value = self.stats_acts_q_val[q_state].get(action, 0)

        self.stats_acts_q_val[q_state][action] = \
            q_value + \
            self.alpha * \
            (getReward(q_state.pacman_pos,
                       q_state.food_pos,
                       q_state.ghosts_pos,
                       q_state.walls) +
             self.gamma *
             max_next_q_values(q_state.pacman_pos, q_state.ghosts_pos,
                               q_state.food_pos, self.stats_acts_q_val,
                               legal, q_state.walls)
             - q_value)

    def getQValue(self, action, state):
        actions_q_val = self.stats_acts_q_val.get(state, 0)
        if actions_q_val is not 0:
            return actions_q_val.get(action, 0)
        else:
            return actions_q_val

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):
        self.updateStatesActionsQValue(self.prev_action,
                                       self.prev_q_state,
                                       state.getLegalPacmanActions())
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg, '-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)
