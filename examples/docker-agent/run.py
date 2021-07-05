"""Implementation of a simple deterministic agent using Docker."""
from train import DQNAgent, DQN

from pommerman import agents
from pommerman.runner import DockerAgentRunner




def main():
    '''Inits and runs a Docker Agent'''
    yyyAgent = DQNAgent(DQN())
    yyyAgent.run()
    #agent = MyAgent()
    #agent.run()


if __name__ == "__main__":
    main()
