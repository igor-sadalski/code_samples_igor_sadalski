from dataclasses import dataclass
from typing import NamedTuple, NewType, Optional, Generator
from copy import deepcopy
from datetime import datetime, timedelta
from itertools import combinations
from collections import deque
from copy import deepcopy

import networkx as nx

import config
from utilities import log_runtime_and_memory

class CountdownIterator:
    '''clean way to keep track of how long the loop has been executing'''
    def __init__(self, start: int):
        self.count = start + 1

    def __iter__(self):
        return self

    def __next__(self) -> bool:
        self.count -= 1
        if self.count < 0:
            return False
        return True


@dataclass
class Clock: #TODO request creation times are clipped only to minutes
    '''class to help keep track of the time during main visualization loop simulation'''
    time: datetime

    def increment(self, min_increment: int = 1):
        '''if you are at a bus stop and need to wait for a new requests
        just imcrement the timer of the clock'''
        self.time += timedelta(minutes=min_increment)

    def set_time(self, new_time: datetime):
        '''replace current time with a new datetime'''
        if isinstance(datetime, new_time):
            self.time = new_time
        else:
            raise TypeError("new time must be an instance of datetime!")
        

class RequestChain(NamedTuple):
    '''imuttable, list of requests that are chained together'''
    requests_chain: list['Request']

    def have_one_value(self) -> bool:
        '''check if request chain is empty'''
        return self.requests_chain == []

    def from_depth(self, depth: int) -> 'Request':
        '''create a specific request chain based on the tree depth'''
        return self.requests_chain[depth]

    def __getitem__(self, index):
        return self.requests_chain[index]
    
#TODO delete this move all dates to inbuild datetime in python
class Date(NamedTuple):
    '''immutable time object, just as we have in our dataset'''
    year: int
    month: int
    day: int


BusID = NewType('BusID', int)
RequestID = NewType('RequestID', int)
NodeID = NewType('NodeID', int)
Utility = NewType('Utility', float)
FilePath = NewType('FilePath', str)

class Time(NamedTuple):
    '''immutable time object, for generative model we consider
    only requested times of pickups'''
    hour: int
    minute: int
    second: int

    # TODO turn off group by minutes not secons

# TODO just use python build in datetime make it compatible with daniel version
class DateTime(NamedTuple):
    '''immutable time object, for generative model we consider
    all dates to be the same'''
    date: Date
    time: Time

class Request(NamedTuple):
    '''imuutable, transformed row of historic data required for our
    baseline algorithm'''
    node_pickup: int
    node_dropoff: int
    pickup_time: datetime  # TODO assume pickup time is same as dropoff time
    # request_creation_time: datetime| None = None #TODO switch from time to datetime
    passengers: int  # TODO change export function to accommodate for this
    id: int

    def get_pickup(self) -> 'PathNode':
        '''get the pickup node as the PathNode'''
        return PathNode(self.node_pickup)

    def get_dropoff(self) -> 'PathNode':
        '''get the dropoff node as the PathNode'''
        return PathNode(self.node_dropoff)

    def get_nodes_values(self) -> tuple[int, int]:
        '''retrieve node_pickup and node_dropoff values as a tuple'''
        return self.node_pickup, self.node_dropoff

RequestID = NewType('RequestId', int)

class PathNode(NamedTuple):
    '''node in a path of a bus as given by requests'''
    node_id: int
    assigned_node_requests: dict[RequestID, tuple[Request, bool]] = {}
    # todo think if you should specif if this si path node or notl

    def insert_request(self, request: Request, is_pickup: bool):
        '''add key of request.id and value of request and whether its a picukp'''
        self.assigned_node_requests[request.id] = (request, is_pickup)

    def delete_request(self, request: Request):
        '''remove request and accompanying metadata (e.g. is_pickup) of this request'''
        self.assigned_node_requests.pop(request.id)

    def get_passanger_change(self) -> int:
        '''iterate over all requests assigned to this node and depending
        on whether its pickup or drop-off add how many people will enter/leave
        the bus'''
        change = 0
        for request, is_pickup in self.assigned_node_requests.values():
            if is_pickup:
                change += request.passengers
            else:
                change -= request.passengers
        return change
    
    def get_all_concluded_request(self) -> list[Request] | None:
        '''assuming this node will be removed return all requests 
        that had drop-off in this node'''
        #TODO in future freeze all request that where dropped off in this node
        return [request for request, is_pickup in self.assigned_node_requests
                        if is_pickup == False] 


@dataclass
class DetailedPath:
    '''planned_node_path with nodes connected by shortests nodes distances;
    DS to keep track of lowest level detailed node path being
    executed by the bus; this is just used for visualization and its
    created at runtime by bus route from bus route objet
    '''
    path: deque[NodeID] = deque([])
    
    # @log_runtime_and_memory
    @classmethod
    def build(cls, planned_node_path: 'PlannedNodePath', curr_pos: NodeID):
        '''out detailed path must be fully rebuild since we can change the
        ordering of the nodes in the planned_node_path; start by connecting
        the node where you stand with the nearest node in the path'''
        detailed_path = cls()
        flags = planned_node_path.planned_node_path
        if curr_pos != flags[0]:
            detailed_path.extend_dq(curr_pos, NodeID(flags[0].node_id)) #TODO add assertion that path makes sense
        for ind in range(len(flags)-1):
            detailed_path.extend_dq(NodeID(flags[ind].node_id), 
                                    NodeID(flags[ind+1].node_id))
        return detailed_path

    def extend_dq(self, curr_node: NodeID, next_node: NodeID):
        '''extend the path with the ID of nodes that form the shortest path 
        between two nodes'''
        self.path.extend(nx.shortest_path(
            config.MAP.graph, curr_node, next_node)[1:])

    # TODO change this to time
    def get_next_node_and_hop_time(self) -> tuple[NodeID, int]:
        '''pop leftmost node (i.e. the current bus node); after poping return
        the new leftmost node where the bus stands; discard the current node'''
        curr_node = self.path.popleft()
        next_node = self.path[0]
        travel_time = config.MAP.get_travel_time(NodeID(curr_node), NodeID(next_node))
        return next_node, travel_time

    def is_empty(self) -> bool:
        '''check if there are any nodes in the current planned path'''
        return bool(not self.path)
    
#TODO create a general class that can do this
#TODO create a metaclass to do this
@dataclass
class BestPath: 
    '''store the best request to swap'''
    curr_utility: Utility
    path: Optional['PlannedNodePath']  # TODO fix how to use this

    def update(self, utility_new_path: Utility, new_path: 'PlannedNodePath'):
        '''we want to maximize utility so switch if new utility is higher'''
        if utility_new_path > self.curr_utility:
            self.curr_utility = utility_new_path
            self.path = new_path

    def unpack(self) -> tuple[Utility, Optional['PlannedNodePath']]:
        return self.curr_utility, self.path


class PlannedNodePath(NamedTuple):
    '''list of nodes in the path of a buss, begining and ending with
    depots, all other nodes must be inserted in between the starting and depot node'''
    #TODO change this to deque!
    planned_node_path: list[PathNode] = [PathNode(config.DEPOT_NODE)] #TODO change this to just path
    #TOOD add the last stop at the request that shows up at some time when the bus need to go back home
    #TODOd change this to path instead of planned node path
    def pop_left(self) -> PathNode:
        '''pop left from planned node path'''
        return self.planned_node_path.pop(0)

    def peek_first(self) -> NodeID:
        '''peek the first value int he planned path'''
        return NodeID(self.planned_node_path[0].node_id)

    def pop(self, request: Request):
        '''identify the node with query request and delete it this request from
        that node'''
        # TODO infuture this can be spedup to O(1) from O(n)
        for path_node in self.planned_node_path:
            if request.id in path_node.assigned_node_requests:
                # we dont care if this is pickup or droop of delte either way
                path_node.assigned_node_requests.pop(request.id)

    # TODO in future actualy once we get rid of initial depot we can insert pickups after the path
    # TODO handle if a feasible insertion doesnt exist...
    def gen_possible_paths(self, request: Request) -> Generator[tuple[Utility, 'PlannedNodePath'], None, None]:
        '''greedy insertion to planned node path iterate over all busses exhaustively 
        try all possible combinations of inserting first pickup, checking feasibility, 
        then inserting dropoff and checking feasibility again, if both are feasible adding 
        such new path to the RV graph, cannot add to the very begining or very end since 
        we star/end in the bus depot!, assume these alawys have initial starting element'''
        best_path = BestPath(Utility(float('-inf')), None)  
        # TODO MUST merge identical nodes!
        for insert_pickup_ind, insert_dropoff_ind in self._combinations_generator():
            candidate = self.deep_copy_to_DS()
            candidate.planned_node_path.insert(insert_pickup_ind, request.get_pickup()) #TODO this doesnt make to much sense
            candidate.planned_node_path.insert(insert_dropoff_ind, request.get_dropoff())
            if candidate.path_is_feasible():
                candidate_util = candidate.get_path_utility()
                best_path.update(candidate_util, candidate)
                yield candidate_util, candidate

    def _combinations_generator(self) -> Generator[tuple[int, int], None, None]:
        '''All possible insertion index combinations for two new values:
        [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 2), (1, 3), (1, 4), (1, 5), ...;
        we start iterating after depot node'''
        yield from combinations(range(1, len(self.planned_node_path) + 2), 2)

    # TODO add suppport to also return the best cadidate!!
    def insert_greedy(self, request: Request) -> tuple[Utility, Optional['PlannedNodePath']]:
        '''iterate over all possible paths; update what is the path with the lowest utilty;
        return that path along its utility'''
        #TODO alternatively make greedy in place!
        best_path = BestPath(Utility(float('-inf')), None)  
        # TODO what to do when system doesn't return anything
        gen = self.gen_possible_paths(request) #TODO move to the trick with the ma and, key , enumerate
        for candidate_util, candidate in gen:
            best_path.update(candidate_util, candidate)
        return best_path.unpack()

    def merge_identical(self, prev: PathNode, curr: PathNode, nxt: PathNode):
        if prev.node_id == curr.node_id:
            pass

    def deep_copy_to_DS(self):
        candidate_list = deepcopy(self.planned_node_path)
        candidate: PlannedNodePath = PlannedNodePath(candidate_list)
        return candidate

    def path_is_feasible(self) -> bool:
        '''simulate the path check if it violates bus capacity
        constrains, iterate over all nodes in the path and check
        how inflow/outflow changes them'''
        num_passengers = 0
        for path_node in self.planned_node_path:
            num_passengers += path_node.get_passanger_change()
            if num_passengers < 0 or num_passengers > config.FLEET.capacity:
                return False
        return True

    # TODO make sure that path is feasible
    def get_path_utility(self) -> Utility:
        '''MUST BE CALLED ON FEASIBLE PATH use the utility function of minimizing the passenger travel time (PTT in paper) 
        thus maximizing  the availabel capacity in each vehicle ove r the 
        time horizon'''
        utility = 0
        num_passengers = 0
        for ind in range(len(self.planned_node_path)-1):
            curr_node = self.planned_node_path[ind]
            next_node = self.planned_node_path[ind+1]
            num_passengers += curr_node.get_passanger_change()
            dist_time = config.MAP.get_travel_time(
                NodeID(curr_node.node_id), NodeID(next_node.node_id))
            utility += num_passengers*dist_time
        return Utility(utility)
    
    def as_list_NodeIDs(self) -> list[NodeID]:
        '''return the planned node path as a list of node IDs'''
        return [NodeID(node.node_id) for node in self.planned_node_path]

    def as_node_list(self) -> list[int]:
        '''return the planned node path as a list of node id and 
        bool info if that was a pickup node or not'''
        return [pth.node_id for pth in self.planned_node_path]
   
class PlannedRequestSequence(NamedTuple):
    '''lsit of all requested allocated ot the funciton; at the begining of the bus routing algorithm
    we dont have any request allocated to the buses; we must have at least one request in the bus;
    otherwise is is just sitting in the depot; ordering of requests doesnt matter'''
    planned_requests: list[Request] = []

    def append(self, request: Request):
        '''append to the request sequence, order doesnt matter'''
        self.planned_requests.append(request)

    def pop(self, request_pos: int) -> Request:
        '''pop request from planned request sequence and return it'''
        return self.planned_requests.pop(request_pos)
    
    def get_as_node_drop_list(self):
        '''retrieve list of request pickup and dropoff nodes (order doesnt matter)
        as a list that can be used to plot the position of the flags; flatten a list
        of tuples to regular list'''
        list_of_tuples = [req.get_nodes_values() for req in self.planned_requests]
        return [item for sublist in list_of_tuples for item in sublist] 

    def remove_requests(self, requests_to_be_removed: list[Request]):
        '''inplace, iterate over requests_to_be_removed and remove the first matchin
        occurence of the thing'''
        for removed_request in requests_to_be_removed:
            self.planned_requests.remove(removed_request)

    def as_list_of_ids(self) -> list[int]:
        '''for plotting visualization plot as list of hashes'''
        return [req.id for req in self.planned_requests]

@dataclass
class BusRoute:
    '''requests and planned node path for a single bus; can be intialized or passed without
    requests and then we only have planned nodes and bus identifier in out system; 
    for reasons of implementing this algorihtm sometimes we dont need to track the requests we 
    just need the paths use None to turn this off
    planned_node_path - just the correctly arranged request pickup/dropoff nodes
    detailed_planned_node_path - planned_node_path + all intermediate nodes that form shortest distances'''
    bus_index: BusID
    planned_requests: PlannedRequestSequence = PlannedRequestSequence()
    planned_node_path: PlannedNodePath = PlannedNodePath()
    # detailed_planned_node_path: DetailedPath = DetailedPath()

    def allocate(self, request: Request) -> Utility:
        '''INPLACE, deepcopy the bus contents and insert request to the planned_node_path;
        supports both instances of BusRoute with and without planned_requests;
        automaticaly checks if the allocation is feasible or not;
        return the utility of the request allocation given the path
        is feasible to return anythin'''
        self.planned_requests.append(request) #TODO if we dont want to append value to this??
        # TODO add suport to check if the allocation is automaticaly feasible
        new_path_util, new_path_with_request = self.planned_node_path.insert_greedy(request)
        if new_path_with_request:
            self.planned_node_path = new_path_with_request #TODO change greedy to inplace and delete this line
            return new_path_util
        else:
            # TODO change this in future
            raise ValueError('allocation of this request is not possible')

    def unallocate(self, request_index: int): #TODO change this to request hash
        '''pop request at request_index from BusRoute and remove its accompanying nodes 
        from the bus planed list of nodes'''
        if self.planned_requests:
            # try to start operating on the reuqests IDs
            popped_request = self.planned_requests.pop(request_index)
            # TODO must identify these requests with their coresponding reuquest IDs
            self.planned_node_path.pop(popped_request)
        else:
            raise ValueError(
                'this instance of bus has no requests despite having nodes')

    def create_detailed_plan(self, curr_pos: NodeID) -> DetailedPath:
        '''build detailed execution path based on the planned node path;
        connect any two adjecent nodes with nodes forming shortests distance;
        start at the current position node and move to the nearest node'''
        return DetailedPath().build(self.planned_node_path, curr_pos)
    
    def remove_node(self, node: NodeID):
        '''INPLACE, check if the node to be removed is the first one in the planned node path;
        if it is remove it; if it is dropoff node remove its accompanying request from
        the planned requests'''
        if node == self.planned_node_path.peek_first():
            remove = self.planned_node_path.pop_left() #TODO change this list to deque
            # req_list = remove.get_all_concluded_request()
            # self.planned_requests.remove_requests(req_list)

class BusesPaths(NamedTuple):
    '''requests and planned paths for all busses; this assumes buses indecies are consequitive
    integers in range from 0 to 'number of busses, and we simply access BusRoute for some bus
    by accessing correct index in the list'''
    theta: list[BusRoute]  # TODO try to chane this to a dictionary where we use bus routes as keys

    def allocate_greedy(self, request: Request) -> None:
        '''append the request to planned requests, and its drop/pickup to planned path
        iterate over each bus_route in the theta; create a candidate which is just the
        deep copy of its contents; for each candidate compute the utility; get the index of
        candidate with the higest utility; then in place actualy allocate the new request to
        the bus with the highest utility'''
        #TODO check if this actually deepcopies the bus contents
        candidates: list[BusRoute] = deepcopy(self.theta)
        best_index, _ = max(enumerate(candidates), key=lambda cand: cand[1].allocate(request))
        selected_bus = self.get_bus(BusID(best_index))
        _ = selected_bus.allocate(request)

    # TODO change this to dict for faster access
    def get_bus(self, vehicle_id: BusID) -> BusRoute:
        '''clear method to access required bus'''
        return self.theta[vehicle_id]

    def create_augmented_path(self, bus_id: BusID, new_route: PlannedNodePath) -> 'BusesPaths':
        '''drop old BusRoute reference and build a new one'''
        cpy = BusesPaths(deepcopy(self.theta))
        cpy.theta[bus_id] = BusRoute(bus_id, new_route)
        return cpy
    # TODO is this the way i should compute the utility for whole sequence?
    # TODO rewrite this cleanly

    def compute_utility(self, *bus_id: BusID) -> Utility:
        '''iterate over theta and sum up the utility for each individual bus path
        in the list'''
        total_theta_utility: Utility = Utility(0)
        iterable = [self.theta[id] for id in bus_id] if bus_id else self.theta
        for bus_route in iterable:
            total_theta_utility += bus_route.planned_node_path.get_path_utility()
        return total_theta_utility

    # TODO think if no requests should i just return an empty list?
    def get_bus_requests(self, bus_id: BusID) -> PlannedRequestSequence:
        '''getter method for planned request of a specific bus'''
        req = self.theta[bus_id].planned_requests
        return req  # there will be always at least one requests in the system

    def get_bus_paths(self, bus_id: BusID, request: Request) -> Generator[tuple[Utility, PlannedNodePath], None, None]:
        '''getter generator of all possible path with their utilites'''
        return self.theta[bus_id].planned_node_path.gen_possible_paths(request)

class PromisingBussesPaths(NamedTuple):
    '''same as big theta or big X in the paper, space of promising/feasible
    actions out buses can execute'''
    actions: list[BusesPaths]