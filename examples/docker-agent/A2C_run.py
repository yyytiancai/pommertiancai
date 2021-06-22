"""Implementation of a simple deterministic agent using Docker."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pommerman import agents
from pommerman.runner import DockerAgentRunner

import random
from A2C.model import *
from pommerman import constants


class MyAgent(DockerAgentRunner):
    '''An example Docker agent class'''

    def __init__(self):
        self.model = A2CNet(gpu=False)
        import os
        if os.path.exists("A2C/convrnn-s.weights"):
            self.model.load_state_dict(torch.load("A2C/convrnn-s.weights", map_location='cpu'))
        self.agent = Leif(self.model)
        self.agent.clear()
        self.agent.debug = True
        self.agent.stochastic = False
        self.directions = [
            constants.Action.Stop, constants.Action.Up, constants.Action.Down, constants.Action.Left,
            constants.Action.Right, constants.Action.Bomb]

    def init_agent(self, id, game_type):
        return self.agent.init_agent(id, game_type)


    def episode_end(self, reward):
        return self.agent.episode_end(reward)

    def shutdown(self):
        return self.agent.shutdown()

    def act(self, observation, action_space):
        return self.directions[self.agent.act(observation, action_space)].value



if __name__ == "__main__":
    agent = MyAgent()
    agent.run()
