import pickle
from collections import defaultdict
from typing import NewType, NamedTuple, List, Generator, Tuple
import dirs
from utilities import  log_runtime_and_memory
import osmnx as ox


# Existing code below...
DistTime = NewType('DistTime', int)
NodeID = NewType('NodeID', int)


class MapNodePath(NamedTuple):
    '''shortest path between two selected nodes including begin and end node'''
    nodes: List[NodeID]

    @classmethod
    def format(cls, nodes: List[int]):
        '''format the nodes into the correct format for the MapNodePath'''
        return cls([NodeID(node) for node in nodes])


def default_factory(): #TODO fix this
    return defaultdict(int)

# TODO save the map as the pickle for easier reads and writes in the future


class Map:

    def __init__(self):
        # TODO change this to the default dict
        self.dir_path_dict: str = 'path_dict.pkl'  # TODO change to specific path type
        self.dir_time_dict: str = 'path_time_dict.pkl'
        self.dir_map_save: str = 'saved_map.pkl'
        self.graph = ox.load_graphml(dirs.GRAPH_STRUCUTRE)
        self.travel_time: dict[NodeID, dict[NodeID, DistTime]] = defaultdict(
            lambda: defaultdict(int))  # TODO fix this
        self.shortest_path: dict[NodeID, dict[NodeID, MapNodePath]] = defaultdict(
            lambda: defaultdict(list))
        
        self._build()

    # =========PUBLIC============
    
    @log_runtime_and_memory
    @classmethod
    def from_save(cls, dir_map: str):
        '''load from .pkl file without rebuilding'''
        new_map = cls()
        with open(dir_map, "rb") as handle:
            loaded_map = pickle.load(handle)
        new_map.travel_time = loaded_map.travel_time
        new_map.shortest_path = loaded_map.shortest_path
        return new_map
    
    # def save(self): #TODO fix and finish this
    #     '''save file to a .pkl format'''
    #     with open(self.dir_map_save, "wb") as handle:
    #         pickle.dump(self, handle)

    def get_travel_time(self, node1: NodeID, node2: NodeID) -> DistTime:
        '''retrive the travel time between two nodes'''
        return DistTime(self.travel_time[node1][node2])

    def get_nodes_forming_shortes_path(self, node1: NodeID, node2: NodeID) -> MapNodePath:
        '''retrive the time sequence of nodeID that form the shortest path in the graph'''
        # TODO replace this with correct method of computing the shortest path between selected nodes
        return self.shortest_path[node1][node2]
    
    # =========PRIVATE============

    def _add_point(self, node_origin: NodeID, node_end: NodeID, travel_time: DistTime, node_path: MapNodePath):
        '''add a point to the map'''
        self.travel_time[node_origin][node_end] = travel_time
        self.shortest_path[node_origin][node_end] = node_path

    def _load_pkl_dict(self, path_name: str) -> Generator[tuple[tuple[int, int], int | list[int]], None, None]:
        with open(path_name, 'rb') as f:
            dictionary = pickle.load(f)
        for key, value in dictionary.items():
            yield key, value
    
    @log_runtime_and_memory
    def _build(self):
        '''initilaize at the begingin of class creation build the map from the save .pkl files'''
        path_dict = self._load_pkl_dict(dirs.MAP_SHORTESTS_PATH)
        time_dict = self._load_pkl_dict(dirs.MAP_TIME)
        for (key_p, path), (key_t, time) in zip(path_dict, time_dict):
            if key_p == key_t:
                self._add_point(NodeID(key_p[0]), NodeID(key_p[1]),
                                DistTime(time), MapNodePath.format(path))  # type: ignore strings automaticaly will be loaded to correct type
            else:
                raise ValueError(
                    'incorrectly saved dictionaries, coresponding keys do not match')
    def _generate_shortests_paths(self):
        pass

    def _generate_shortests_times(self):
        pass

# mapa = Map()
# map_quicker_read = Map.from_save('saved_map.pkl')
