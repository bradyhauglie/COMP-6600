# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from backend import ReplayMemory

import nn
import model
import backend
import gridworld


import random,util,math
import numpy as np
import copy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        
        self.Q = util.Counter()
       

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # Check if the state is in the Q-values dictionary
        if state not in self.Q:
            # If not, initialize it with an empty dictionary
            self.Q[state] = util.Counter()
        # Check if the action is in the Q-values for the state
        if action not in self.Q[state]:
            # If not, initialize it to 0.0
            self.Q[state][action] = 0.0
        # Return the Q-value for the state-action pair
        return self.Q[state][action]
    

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # Initialize the maximum value to negative infinity
        maxValue = float('-inf')
        # Get the legal actions for the current state
        legalActions = self.getLegalActions(state)
        # If there are no legal actions, return 0.0
        if not legalActions:
            return 0.0
        # Iterate over all legal actions
        for action in legalActions:
            # Get the Q-value for the state-action pair
            qValue = self.getQValue(state, action)
            # Update the maximum value if necessary
            maxValue = max(maxValue, qValue)
        # Return the maximum value found
        return maxValue

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Initialize the best action and its value
        bestAction = None
        bestValue = float('-inf')
        # Get the legal actions for the current state
        legalActions = self.getLegalActions(state)
        # If there are no legal actions, return None
        if not legalActions:
            return None
        # Iterate over all legal actions
        for action in legalActions:
            # Get the Q-value for the state-action pair
            qValue = self.getQValue(state, action)
            # Update the best action if necessary
            if qValue > bestValue:
                bestValue = qValue
                bestAction = action
        # Return the best action
        return bestAction

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # Check if there are legal actions available
        if legalActions:
            # With probability epsilon, choose a random action
            if util.flipCoin(self.epsilon):
                action = random.choice(legalActions)
            # Otherwise, choose the best action according to Q-values
            else:
                action = self.computeActionFromQValues(state)
        # If there are no legal actions, return None
        else:
            action = None
        
        return action

    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # Get the current Q-value for the state-action pair
        currentQValue = self.getQValue(state, action)
        # Get the maximum Q-value for the next state
        maxNextQValue = self.computeValueFromQValues(nextState)
        # Calculate the new Q-value using the Q-learning update rule
        newQValue = (1 - self.alpha) * currentQValue + self.alpha * (reward + self.discount * maxNextQValue)
        # Update the Q-value for the state-action pair
        self.Q[state][action] = newQValue
        

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # Get the feature vector for the state-action pair
        features = self.featExtractor.getFeatures(state, action)
        # Initialize the Q-value to 0
        qValue = 0.0
        # Iterate over all features
        for feature, value in features.items():
            # Calculate the Q-value as the dot product of weights and features
            qValue += self.weights[feature] * value
        # Return the Q-value
        return qValue

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # Get the feature vector for the state-action pair
        features = self.featExtractor.getFeatures(state, action)
        # Get the maximum Q-value for the next state
        maxNextQValue = self.computeValueFromQValues(nextState)
        # Calculate the difference between the observed reward and the estimated Q-value
        difference = (reward + self.discount * maxNextQValue) - self.getQValue(state, action)
        # Update the weights using the feature vector and the difference
        for feature, value in features.items():
            # Update the weight for each feature
            self.weights[feature] += self.alpha * difference * value
        

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
            # Print the learned weights
            print("Learned Weights:")
            for feature, weight in self.weights.items():
                print(f"{feature}: {weight}")
