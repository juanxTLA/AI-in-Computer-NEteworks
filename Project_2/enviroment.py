import numpy as np
import random

class Environment:
    def __init__(self, numUsers, numChannels, attemptProbability = 1.0):
        self.attemptProbability = attemptProbability
        self.numUsers = numUsers
        self.numChannels = numChannels
        self.reward = 1

        #action for each channel
        self.actions = np.arange(self.numChannels)
        self.userActions = np.zeros([self.numUsers], np.int32)
        self.userObs = np.zeros([self.numUsers], np.int32)

    def sample(self):
        x = np.random.choice(self.actions, size=self.numUsers)
        return x

    def step(self, action):
        channelAllocation = np.zeros([self.numChannels + 1], np.int32)
        res = []
        reward = np.zeros([self.numUsers])

        j = 0
        for i in action:
            p = np.random.uniform(0,1)

            if p <= self.attemptProbability:
                self.userActions[j] = i
                
            channelAllocation[i] += 1
            j += 1

        #if more than one user is trying to access channel then unable to use channel
        for i in range(1, len(channelAllocation)):
            if(channelAllocation[i] > 1):
                channelAllocation[i] = 0
        

        for i in range(len(action)):
            self.userObs[i] = channelAllocation[self.userActions[i]]

            if self.userActions[i] == 0:
                self.userObs[i] = 0
            
            if self.userObs[i] == 1:
                reward[i] = 1
            
            res.append((self.userObs[i], reward[i]))

        channelCapacity = channelAllocation[1:]
        channelCapacity = 1 - channelCapacity

        res.append(channelCapacity)

        return res