from copy import deepcopy
from typing import NamedTuple
from heapq import heappush

from config import *
from supporting_DS import PromisingBussesPaths, Request, BusesPaths
from RV_Graph import *
from VV_Graph import *


class HeapNode(NamedTuple):
    '''node used by the feasible action to quickly select n largest
    actions and create promining actions from them (as in paper)'''
    utility: Utility
    theta: BusesPaths

    # TODO invert heap opearions to automaticaly keep maximum element on the otp
    # def __lt__(self, other: 'HeapNode') -> bool:
    #     '''invert the comparison for max heap'''
    #     return self.utility > other.utility


class FeasibleActionsStore(NamedTuple):
    '''heap/priority queue of all feasible actions, consisting
    of utlity as priority and given BusesPaths as values, allows
    for quick retrieval of top items when we want to receive
    promising actions'''
    heap: list[HeapNode] = []

    def push(self, utility: Utility, theta: BusesPaths):
        '''push node onto the feasible actions heap'''
        node = HeapNode(utility, theta)
        heappush(self.heap, node)

    def get_k_biggest(self) -> PromisingBussesPaths:
        '''select k most promising actions from all feasible actions
        in paper same as building X_t'''
        lst = [heapnode.theta for heapnode in nlargest(
            config.K_MAX, self.heap)]
        return PromisingBussesPaths(lst)


class PromisingActions(NamedTuple):
    '''implementation of algo 1 used to generate promising actions during the 
    expansion phase of building the MCTS tree'''
    rv_graph: RVGraph
    vv_graph: VVGraph
    request: Request
    theta: BusesPaths
    heap: FeasibleActionsStore = FeasibleActionsStore()

    # TODO change this to a simple funciton or a __post__init__ expresion
    def generate(self) -> PromisingBussesPaths:
        '''notation for variables, while confusing is used from the 
        paper; '''
        for vehicle_id, er_ij_path in self.rv_graph.edge_iterator():
            updated_theta_ij = self.theta.create_augmented_path(
                BusID(vehicle_id), er_ij_path)
            u_x = er_ij_path.get_path_utility()
            self.heap.push(u_x, updated_theta_ij)
            vv_copy = deepcopy(self.vv_graph)
            vv_copy.delete_vertex(BusID(vehicle_id))
            while vv_copy.E_VV:
                vv_edge = self.vv_graph.arg_max()
                u_x += vv_edge.utility_change
                for bus_id, bus_route in vv_edge.get_bus_index_and_path():
                    updated_theta_ij = self.theta.create_augmented_path(
                        BusID(bus_id), bus_route)
                self.heap.push(Utility(u_x), updated_theta_ij)
                vv_copy.delete_vertex(BusID(vv_edge.bus_m_route.bus_index),
                                      BusID(vv_edge.bus_n_route.bus_index))

        return self.heap.get_k_biggest()

