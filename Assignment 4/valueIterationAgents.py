# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        "*** YOUR CODE HERE ***"
        # Initialize the values for all states to 0
        states = self.mdp.getStates()
        for state in states:
            self.values[state] = 0
        # Perform value iteration for the specified number of iterations
        for i in range(self.iterations):
            # Create a copy of the current values to update
            newValues = self.values.copy()
            for state in states:
                # If the state is terminal, skip it
                if self.mdp.isTerminal(state):
                    continue
                # Initialize the maximum Q-value for this state
                maxQValue = float('-inf')
                # Iterate over all possible actions for this state
                for action in self.mdp.getPossibleActions(state):
                    # Calculate the Q-value for this action
                    qValue = self.computeQValueFromValues(state, action)
                    # Update the maximum Q-value if necessary
                    maxQValue = max(maxQValue, qValue)
                # Update the value for this state using the maximum Q-value
                newValues[state] = maxQValue
            # Update the values with the new values calculated in this iteration
            self.values = newValues
        return self.values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Initialize the Q-value to 0
        qValue = 0
        # Iterate over all possible next states and their probabilities
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            # Calculate the reward for this action and next state
            reward = self.mdp.getReward(state, action, nextState)
            # Update the Q-value using the Bellman equation
            qValue += prob * (reward + self.discount * self.values[nextState])
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Initialize the best action and its value
        bestAction = None
        bestValue = float('-inf')
        # Iterate over all possible actions for this state
        for action in self.mdp.getPossibleActions(state):
            # Calculate the Q-value for this action
            qValue = self.computeQValueFromValues(state, action)
            # Update the best action if necessary
            if qValue > bestValue:
                bestValue = qValue
                bestAction = action
        # Return the best action found
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
