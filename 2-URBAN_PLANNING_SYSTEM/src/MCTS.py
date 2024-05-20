from abc import ABC, abstractmethod
from collections import Counter
import math as m
from typing import NamedTuple, NewType, Optional
from heapq import heappush

from supporting_DS import *
from generative_model import *  # TODO change this in the future
from algo1 import *
from RV_Graph import *
from VV_Graph import *
import config
from generative_model import *
from interfaces import MCTS
from utilities import  log_runtime_and_memory


@dataclass
class UCB:
    '''store all info for computing ucb val'''
    val: float = 0  # the actual value used by the selection algo

    avg_reward: float = 0
    tuning_param: float = config.MCTS_TUNING_PARAM


@dataclass
class MCTSHeapNode:
    '''node of an MCTS tree implemented as a heap which allows
    for a quick retrieval of values from the tree'''
    ucb_score: float
    node: 'MCNode'


@dataclass
class MCNode:
    '''node of an MCTS tree that supports findings its UCB value and
    updating how many times it has been visited'''
    request: Request  # TODO: do i need to store info for these guys
    theta: BusesPaths  # TODO: do i need to store info for these guys
    # self reference with forward refernce , at root not defined
    parent: 'Optional[MCNode]' = None
    children: list[MCTSHeapNode | None] = []  # keep children as a heap
    visitis: int = 0  # select will automaticaly update to 1
    ucb: UCB = UCB()

    def update_UCB_val(self,):
        '''use standard Upper Condifence Bound equation to compute the
        value inside of the node'''
        # you start at root so value you have here doesnt matter
        nom = self.parent.visitis if self.parent else 1
        denom = self.visitis
        freq_term = (m.sqrt(m.log(nom) / denom))
        self.ucb.val = self.ucb.avg_reward + self.ucb.tuning_param * freq_term

    def update_visits(self):
        '''update should always happen in the forward pass
        during the select action'''
        self.visitis += 1

    def select_best_child(self) -> 'MCNode':
        '''retrieve root child from children heap'''
        if self.children[0]:  # 0th element is max
            return self.children[0].node
        else:
            raise ValueError('There are no children in this list')

    def append_child(self, node: 'MCNode'):
        '''convert a node to a heap node and push it onto the heap
        of children nodes'''
        # TODO should i initilaize this with zero value?
        heap_node = MCTSHeapNode(0, node)
        heappush(self.children, heap_node)


class MCTree(MCTS):  # TODO create ABS and python interfaces to make substicuting techniques for evaluating these tress even quicker

    def __init__(self, start_paths: BusesPaths, start_requests: Request, sampled_requests: RequestChain):
        self.sampled_requets = sampled_requests
        self.root: MCNode = MCNode(start_requests, start_paths)
        # TODO on highlevel double check if there isn't anything else we would need to pass here
        self.rv_graph = RVGraph(start_requests, start_paths)
        self.vv_graph = VVGraph(start_paths)
        self.promising_actions = PromisingActions(self.rv_graph, self.vv_graph, start_requests, start_paths).generate(
        ).actions  # TODO fix this to more cleaer interface

    def select(self, node: MCNode | None = None, depth=0) -> tuple[int, MCNode]:
        '''starting at root recursively select child with highest
        UCB value; at leaf return this child'''
        node = node if node else self.root
        node.update_visits()  # TODO make sure this is executed on a global instance of node
        if node.children:
            return self.select(node.select_best_child(), depth+1)
        else:
            return depth, node

    def expand(self, selected_node: MCNode, next_request: Request):
        '''starting from selected node compute promising actions based on buses paths and
        request in this node using algo 1; append new generated nodes as children'''
        for promising_action in self.promising_actions:
            new_node = MCNode(next_request, promising_action)
            selected_node.append_child(new_node)

    def rollout(self, selected_node: MCNode, requests: RequestChain, theta: BusesPaths):
        '''recurse down while we have requests (this should be equal to tree depth); when you
        run of out requests compute the utility of of the BusesPaths; updated rolloued out
        node with this values'''
        new_theta = RVGraph(selected_node.request, theta).greedy_assignment()
        if requests.have_one_value():
            # TODO should i just take the last value at the bottom?
            # TODO convert this to float
            selected_node.ucb.avg_reward = new_theta.compute_utility().val
        else:
            # no risk we pass an empty request
            self.rollout(selected_node, requests[1:], new_theta)

    def backpropagate(self, selected_node: MCNode):
        '''recurse up the tree and update average of each
        tree node in the path; no need to update root as we will always select
        it anyways; this will be initialy called on a already rolled out node
        to backprop its value through the whole stack'''
        if selected_node.parent:
            selected_node.update_UCB_val()
            self.backpropagate(selected_node.parent)


@dataclass
class ActionValue:
    avg_reward: float
    busses_paths: BusesPaths
    n: int = 0  # TODO question is whether each node in the initial layer will be visited at least once

    def update(self, new_val: float):
        '''update given action values pair with new values'''
        self.avg_reward = (self.avg_reward * self.n + new_val) / (self.n + 1)
        self.n += 1


class ActionValues(NamedTuple):
    '''DS to update and keep track of the best actions/buses paths in our applications'''
    # TODO how to get rid of instance when its none...
    action_values: list[ActionValue | None] = []

    def _build(self, tree: MCTree):
        '''initialize a list of action nodes from the children found at the root of a tree'''
        for heap_node in tree.root.children:
            if heap_node:  # TODO this is to complicated change this
                self.action_values.append(ActionValue(0, heap_node.node.theta))

    def get_best_action(self) -> BusesPaths:
        '''return the action with the highest average reward'''
        return max(self.action_value, key=lambda x: x.avg_reward).busses_paths

    def update(self, root: MCTree):
        '''iterate over first expanded action from the root; extract their values and average them with
        the values at the coresponding positions in the action_value list'''
        if not self.action_values:
            self.action_values.build(root)
        for pos, heap in enumerate(root.root.children):
            if heap:
                # TODO fix funky stuff going on with Nones
                self.action_values[pos].update(heap.node.ucb.avg_reward)


class MCForest:
    '''compute predicted utility of each actions we can take from our present state;
    in parallel evaluate trees and update predicted action utilites; in the end take the action
    with highest present and future utlity'''

    def __init__(self, start_paths: BusesPaths, starting_request: Request, historic_data: HistoricData):
        self.start_request = starting_request
        self.start_paths = start_paths
        self.generative_model: GenerativeModel = GenerativeModel(historic_data)
        self.action_values: ActionValues = []

    @log_runtime_and_memory
    def _evaluate_tree(self) -> MCTree:  # TODO to long make this function shorter
        '''perform multiple iterations of selection -> expansion -> rollout -> backprop;
        move the dummy pointer from root to selected node; expand that node; '''
        # TODO each time we want to sample new requests
        requests = self.generative_model.sample_request_from_the_bank()
        root = dummy = MCTree(self.start_paths, self.start_request, requests)
        for _ in range(config.MCTS_ITERATIONS):
            # TODO make sure the pointer is walking down
            dummy = root
            depth, selected_node = dummy.select()
            dummy.expand(selected_node, requests.from_depth(depth))
            for child_node in selected_node.children:
                if child_node:
                    child_node = child_node.node
                    # TODO should this depth be or not be included in the expanded requests?
                    dummy.rollout(
                        child_node, requests[depth:], child_node.theta)
                    dummy.backpropagate(child_node)
        return root

    @log_runtime_and_memory
    def get_best_action(self) -> BusesPaths:
        '''in parralel evaluate multiple MCTS for each one update action utitlies and
        return action with the highest evaluated score; '''
        # TODO implement proper parallelization
        for _ in range(config.MCTS_TREES):
            root = self._evaluate_tree()
            self.action_values.update(root)
        return self.action_values.get_best_action()



# bus_route = BusRoute(BusID(0), planned_node_path, planned_requests)
# cur_paths = BusesPaths()
# cur_request = unseen_new_requests.requests_chain[0]
# for cur_request in unseen_new_requests.requests_chain:
#     bus_route = BusRoute(BusID(0), PlannedRequestSequence([cur_request]))
#     mc_forest = MCForest(cur_paths, cur_request, historic_data)
#     cur_paths = mc_forest.get_best_action()
