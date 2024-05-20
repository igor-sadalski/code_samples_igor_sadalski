
from copy import deepcopy
from dataclasses import dataclass

from typing import NamedTuple, NewType, TypeAlias, Generator
from supporting_DS import Request, BusRoute, Utility, BusID, BusesPaths, PlannedNodePath
import config
from heapq import heapify
#TODO start working on unit testing for all the functions in the stack

@dataclass
class VVEdge:
    '''edge of vvgraph defined in this way to support
    operations required by the algorithm 1'''
    bus_m_route: BusRoute
    bus_n_route: BusRoute
    utility_change: Utility

    def update(self, bus_m: BusRoute, bus_n: BusRoute, new_utility: Utility) -> None:
        '''we want to maximize utility so switch if new utility is higher'''
        if new_utility > self.utility_change:
            self.bus_m_route = bus_m
            self.bus_n_route = bus_n
            self.utility_change = new_utility
    
    def get_bus_index_and_path(self) -> Generator[tuple[int, PlannedNodePath], None, None]:
        '''clean way to return bus index and planned path for bus_m and bus_n 
        in our vv_edge'''
        yield from [(self.bus_m_route.bus_index, self.bus_m_route.planned_node_path),
                     (self.bus_m_route.bus_index, self.bus_m_route.planned_node_path)]   


@dataclass
class AdjecenyList:
    '''representation of our graph'''
    ll: dict[BusID, dict[BusID, VVEdge]]

    def insert_edge(self, edge: VVEdge):
        '''insert VVEdge into VVGraph'''
        self.ll[BusID(edge.bus_m_route.bus_index)][BusID(
            edge.bus_n_route.bus_index)] = edge

    def extract_values(self, nested_dict: 'AdjecenyList') -> Generator[VVEdge, None, None]:
        '''recursively find all values in anested adjecency list
        implemented as dictionary of dictionaries'''
        for value in nested_dict.ll.values():
            if isinstance(value, dict):
                yield from self.extract_values(value)
            else:
                yield value

# TODO convert atomic types to classes for easier identification of types


class VVGraph:
    '''represented the graph as an adjeceny list with
    dictionaries, support some operations required by algo 1'''

    def __init__(self, buses_paths: BusesPaths):
        self.E_VV = AdjecenyList({})
        self._build(buses_paths)

    def _build(self, buses_paths: BusesPaths):
        '''iterate over all possible bus pair for each pair select the best request to swap'''
        bus_gen = ((BusID(m), BusID(n)) for m in range(config.FLEET.num_busses)
                   for n in range(config.FLEET.num_busses)
                   if m != n)
        for m, n in bus_gen:
            initial_utility: Utility = buses_paths.compute_utility(m, n)
            vv_edge = self._create_edge(buses_paths.get_bus(m), buses_paths.get_bus(n), initial_utility)
            if vv_edge:
                self.E_VV.insert_edge(vv_edge)

    # TODO add request index here

    def _create_edge(self, bus_m: BusRoute, bus_n: BusRoute, initial_utility: Utility) -> VVEdge | None:
        '''iterate over all requests allocated to bus m; at each iteration try to unallocate given request
        from the bus and allocate it to the other bus.'''
        edge = None
        if not bus_m.planned_requests:
            return None
        for selected_request_index, selected_request in enumerate(bus_m.planned_requests.planned_requests): #TODO fix this allocation its dumn
            cpy_m = deepcopy(bus_m)
            cpy_m.unallocate(selected_request_index)
            cpy_n = deepcopy(bus_n)
            # if allocation is not feasible return None
            cpy_n.allocate(selected_request)
            if cpy_m and cpy_n:
                new_utility = Utility(cpy_m.planned_node_path.get_path_utility() +
                                      cpy_n.planned_node_path.get_path_utility() -
                                      initial_utility)
                if not edge:
                    edge = VVEdge(cpy_m, cpy_n, new_utility)
                else:
                    edge.update(cpy_n, cpy_m, new_utility)
        return edge

    # TODO: replace this with simple double for loop and swaping value

    def arg_max(self) -> VVEdge:
        '''find edge with minimum utility, flatten graph by extracting
        only edges in the graph, create heap from them and retrieve
        root value from the heap'''  # TODO check minimum maximum utility
        out = list(self.E_VV.extract_values(self.E_VV))
        heapify(out)
        return out[0]

    def delete_vertex(self, *vehicles: BusID):
        '''remove dict row where each vehicle in vehicles is the key then iterate over 
        all other keys and remove the inner key which is equal to each vehicle in vehicles;
        support deletion of multiple verticies in the graph'''
        for vehicle in vehicles:
            self.E_VV.ll.pop(vehicle, None)
            for vehicle_key in self.E_VV.ll:  # type: ignore
                self.E_VV.ll[vehicle_key].pop(vehicle, None)
