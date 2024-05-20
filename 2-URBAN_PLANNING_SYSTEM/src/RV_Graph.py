from copy import deepcopy
from typing import NamedTuple, Generator
import config
from heapq import heapify, heappush, nlargest
from supporting_DS import Utility, PlannedNodePath, BusID, BusesPaths, Request, BusRoute


class RVEdge(NamedTuple):
    '''edge of RV graph'''
    utility: Utility
    planned_node_path: PlannedNodePath


class RVGraph:
    '''represent the graph as an adjeceny list with dictionaries'''

    def __init__(self, request: Request, theta: BusesPaths):
        self.request = request
        self.theta = theta
        # keeping this as a heap helps us for the MCTS implementation during selection
        # TODO think about how to define default case for this
        self.E_RV: dict[BusID, list[RVEdge]] = {
            BusID(i): [] for i in range(config.FLEET.num_busses)}
        self._build()

    # =========PUBLIC============

    def insert_edge(self, vehicle: BusID, path: PlannedNodePath):
        '''insert RVEdge into RVGraph'''
        # TODO MAX VS MIN UTILITY BE CARFUL AND FIX WHAT YOU ACTUALY WANT TO PUSH
        node = RVEdge(path.get_path_utility(), path)
        heappush(self.E_RV[vehicle], node)

    def edge_iterator(self) -> Generator[tuple[int, PlannedNodePath], None, None]:
        '''generator to iterate over all edges in the RV graph,
        vehicle_id coresponds to i'''
        for vehicle_id in range(config.FLEET.capacity):
            for heapnode in self.E_RV[BusID(vehicle_id)]:
                yield vehicle_id, heapnode.planned_node_path

    def greedy_assignment(self) -> BusesPaths:
        '''choose the edg ewith the highest myopic utility do not incorporate
        swapping requests into rollout policy'''
        out = BusesPaths()
        for vehicle_id in self.E_RV:
            bus_path = BusRoute(vehicle_id,
                                self._get_max_path(vehicle_id))
            # TODO you need to push with negative priority to keep the stack in place
            out.theta.append(bus_path)
        return out

    # =========PRIVATE============

    def _get_max_path(self, vehicle: BusID) -> PlannedNodePath:
        return self.E_RV[vehicle][1].planned_node_path

    def _build(self):
        '''iterate over all busses exhaustively try all possible combinations
        of inserting first pickup, checking feasibility, then inserting dropoff
        and checking feasibility again, if both are feasible adding such new path
        to the RV graph'''
        for vehicle_id in range(config.FLEET.num_busses):
            generator = self.theta.get_bus_paths(
                BusID(vehicle_id), self.request)
            self.E_RV[BusID(vehicle_id)] = [RVEdge(utility, path)
                                            for utility, path in generator]