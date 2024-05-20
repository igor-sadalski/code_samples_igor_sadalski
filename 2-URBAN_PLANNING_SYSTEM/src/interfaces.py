from abc import ABC, abstractmethod

from typing import Optional
from supporting_DS import Request, RequestChain, BusesPaths
from MCTS import MCNode

class MCTS(ABC):
    '''interface for designing different MCTS techniques'''

    @abstractmethod
    def select(self, node: MCNode | None = None, depth=0) -> tuple[int, MCNode]:
        '''iterate over the tree and return the node with the highest UCB value'''
    
    @abstractmethod
    def expand(self, selected_node: MCNode, next_request: Request):
        '''expand the selected node by adding new children nodes'''
    
    @abstractmethod
    def rollout(self, selected_node: MCNode, requests: RequestChain, theta: BusesPaths):
        '''simulate the rollout of the tree to the bottom'''

    @abstractmethod
    def backpropagate(self, selected_node: MCNode):
        '''backpropagate the values from the bottom of the tree to the top'''



class BusRoute:
    pass