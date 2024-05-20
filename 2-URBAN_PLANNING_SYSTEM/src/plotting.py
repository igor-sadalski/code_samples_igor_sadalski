import pandas as pd
import osmnx as ox
import networkx as nx
import matplotlib.axes
import matplotlib.pyplot as plt

from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque

import config
from requests_simulator import Data
from supporting_DS import NodeID, Request, BusRoute, BusesPaths,BusID, Clock, CountdownIterator, DetailedPath
from plotting_utils import Annotate, GifCreator, PlotMetadata
from utilities import log_runtime_and_memory


class VisualSimulator:
    '''for some off reason i cannot debug we cannot simply add and remove 
    annotate values to the plt graph, since each time we call '''
    def __init__(self, data: Data, start_date: datetime):
        self.filenames: list[str] = []
        self.dt = data  # TODO came up with better name for this
        self.dt.reset_simulator()
        self.bus_route = BusRoute(BusID(0)) #in future integrate this with bus route to start the model
        self.path = DetailedPath()
        # this will be planned_requests in the future attribute of the BusClass
        self.old_requests = []
        self.clock = Clock(start_date)
        self.plot_metadata = PlotMetadata(clock = self.clock, bus_path = self.bus_route)
        self.gif = GifCreator(self.plot_metadata)
        self.sim_req_limit = CountdownIterator(30)  # TODO add this to config
        self.gif.clear_directory()

    @log_runtime_and_memory
    def run_simulation(self):
        '''while there are future requests in the simulator pull these requests and based 
        on them update bus route or move bus route to the next node in the current planned path;
        once there check again if there are any new requests we could use to update out path'''
        curr_pos = NodeID(config.DEPOT_NODE)  # this is the position of the bus
        while self.dt.are_any_requests() and next(self.sim_req_limit):
            self.plot_metadata.update(current_bus_node=curr_pos)
            while self.dt.is_new_request(self.clock.time):
                new_request = self.dt.get_next()
                self.plot_metadata.update(request_data=new_request)
                self.bus_route.allocate(new_request)
                self.path = self.bus_route.create_detailed_plan(curr_pos) #TODO this shlud be part of planned bus routes 
                self._visualize_new_request(curr_pos, new_request) #TODO plot historic requests before appending to them
            if self.path.is_empty():
                self.clock.increment()
            else:
                self.bus_route.remove_node(curr_pos)
                curr_pos, travel_time = self.path.get_next_node_and_hop_time()
                self._visualize_bus_node_hop(curr_pos, new_request)
                self.clock.increment(travel_time)
        self.gif.generate_gif()
        print('====CONCLUDED======')

    def _visualize_new_request(self, curr_pos: NodeID, new_request: Request):
        '''add new request to the map; update path; generate new frame for the gif'''
        ax, annotate = self._initialize_plot()
        annotate.bus_move(curr_pos)
        annotate.historic_requests(self.bus_route.planned_node_path.as_list_NodeIDs())
        annotate.new_request(NodeID(new_request.node_pickup), NodeID(new_request.node_dropoff))
        self.gif.add_frame(ax, self.path, new_request)

    def _visualize_bus_node_hop(self, curr_pos: NodeID, new_request: Request):
        '''move bus to the next node in its node path; based on elapsed time
        update time it took the bus to move; generate new frame for the gif;
        return the time it took the bus to perform the hop'''
        ax, annotate = self._initialize_plot()
        annotate.bus_move(curr_pos)
        annotate.historic_requests(self.bus_route.planned_node_path.as_list_NodeIDs())
        self.gif.add_frame(ax, self.path, new_request)

    def _initialize_plot(self) -> tuple[matplotlib.axes.Axes, Annotate]:
        '''create canvas for analysing our system and instatiate Annotate class
        object'''
        _, ax = ox.plot_graph(config.MAP.graph, node_size=1, node_color='w', edge_linewidth=0.5,
                              node_alpha=0.8, show=False)
        annotate = Annotate(ax)
        return ax, annotate


date_string = "2022-06-02 19:01:00" #this need to be fine tuned so that simulator wont get lost
date = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")

dt = Data()
vs = VisualSimulator(dt, date)
vs.run_simulation()
