from map import Map
from typing import NamedTuple 
#TODO carefule with circular imports
import pandas as pd
import osmnx as ox

import dirs

class BusFleet(NamedTuple):
    '''configuration class for the bus fleet'''
    num_busses: int = 1
    capacity: int = 4

# Alston
depot_latitudes = [42.3614251]
depot_longitudes = [-71.1283633]

K_MAX = 10
FLEET = BusFleet()
MCTS_DEPTH = 5  
MCTS_ITERATIONS = 1000
N_CHAINS = 25
MCTS_TUNING_PARAM = 1
SAMPLED_BANK_SIZE = 10000
MCTS_TREES = 256
MAP = Map() #TODO switch to only using the loaded osmnx graph in networkx.MultiDiGraph format
DEPOT_NODE: int = ox.nearest_nodes(MAP.graph, depot_longitudes, depot_latitudes)[0] #TODO get rid of the list convert to simple int/NodeID

FPS = 2000

#======XLSX FILES=======
TIME_RANGE_START = pd.to_datetime("2022-04-28 15:00:00")
HISTORY_END_TIME = pd.to_datetime("2022-06-02 15:00:00")
TIME_RANGE_END = pd.to_datetime("2022-06-23 23:59:59") #this date may simply not exist

