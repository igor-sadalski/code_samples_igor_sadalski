import osmnx as ox
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import shutil
import matplotlib.axes
import matplotlib.image as image
from datetime import datetime

import os
import itertools
from dataclasses import dataclass
from typing import Optional

import config
import dirs
from supporting_DS import NodeID, DetailedPath, BusRoute, FilePath, Request, BusesPaths, Clock


# TODO add classes that will improve plotting the informations here on the default graph
@dataclass
class PlotMetadata:
    '''class to log data we want to display next to our graph
    in case given value is not used it defaults to None;
    plot metadata is passed at the creation of the gifcreator;
    and updating plot metadatacontents will be automaticaly passed
    to the gifcreator class'''
    clock: Optional['Clock'] = None
    current_bus_node: Optional[NodeID] = None
    intra_node_travel_time: Optional[int] = None
    request_data: Optional[Request] = None
    buses_paths: Optional[BusesPaths] = None
    bus_path: Optional[BusRoute] = None

    def update(self, **kwargs):
        '''setter method for the class'''
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(
                    'you passed and incorrect arguments to the function')

    def get_time(self) -> datetime:
        '''getter method for time attibute'''
        return self.clock.time

    #TODO create a smart way to disply these information, JSON based
    def generate_contents_text(self) -> str:
        '''generate string that will be displayed next tso each frame of
        simulation'''
        # TODO make this more interactive
        text = f'''
                time: {self.clock.time} \n
                planned_requests: {self.bus_path.planned_requests.as_list_of_ids()} \n
                planned_nodes: {self.bus_path.planned_node_path.as_node_list()} \n
                '''
        if self.request_data:
            text += f'''
                request_pickup: {self.request_data.node_pickup} \n
                request_dropoff: {self.request_data.node_dropoff} \n
                request_creation: {self.request_data.pickup_time} \n
                '''
        if self.current_bus_node:
            text += f'''
                current_bus_node: {self.current_bus_node} \n
                '''
        return text


@dataclass
class Annotate:
    '''class to add customised annotations of the map'''
    ax: matplotlib.axes.Axes

    def routing_start(self):
        '''add starting depot image'''
        self._add_annotation(config.DEPOT_NODE, dirs.BUS_ICON)

    def bus_move(self, cur_pos: NodeID):
        '''move bus to the next node in its node path; based on elapsed time
        update time it took the bus to move'''
        self._add_annotation(cur_pos, dirs.BUS_ICON)

    def historic_requests(self, requests_nodes: list[NodeID]):
        '''insert all the requests currently assigned to the bus as flags'''
        for request_node in requests_nodes:
            self._add_annotation(request_node, dirs.FLAG_ICON)

    def new_request(self, request_pickup: NodeID, request_dropoff: NodeID):
        '''add a new request differentiating between a drop off and pickup
        map markers'''
        self._add_annotation(request_pickup, dirs.PASSENGER_ICON)
        self._add_annotation(request_dropoff, dirs.MARKER_ICON)

    def routing_end(self):
        '''add final depot image'''
        self._add_annotation(config.DEPOT_NODE, dirs.BUS_ICON)

    def _add_annotation(self, node_id: NodeID, icon_path: FilePath):
        '''simply insert annotation image at the specified coordinates'''
        node = config.MAP.graph.nodes[node_id]
        node_latitude = node['y']
        node_longitude = node['x']
        icon = image.imread(icon_path)
        imagebox = OffsetImage(icon, zoom=0.2)
        ab = AnnotationBbox(
            imagebox, (node_longitude, node_latitude), frameon=False)
        self.ax.add_artist(ab)


class GifCreator:
    def __init__(self, plot_metadata: 'PlotMetadata'):
        self.plot_metadata = plot_metadata
        self.filenames: list[str] = []
        self.frame_num = itertools.count()

    def generate_gif(self):
        '''iterate over all generated .png images convert them to image generator
        path and then save them as gif; remove all .png images used to create the 
        gif'''
        images = []
        for filename in self.filenames:  # TODO add context manager here to make it quicker
            images.append(Image.open(filename))
        images[0].save('gifs/output.gif', save_all=True,
                       append_images=images[1:], duration=config.FPS, loop=0)
        for filename in set(self.filenames):
            os.remove(filename)

    def add_frame(self, ax: matplotlib.axes.Axes, path: 'DetailedPath', request: Request):
        '''set gif title and metadata description, plot the newest route, as already has
        all required annotations overlayed on it; assume that there is always one request at 
        the time'''
        ax.set_title('Time:' + str(self.plot_metadata.get_time()))
        text = self.plot_metadata.generate_contents_text()
        ax.text(1.05, 1, text, ha='left', va='top', transform=ax.transAxes)
        filepath = self.fetch_frame_filepath()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig1, _ = ox.plot_graph_route(config.MAP.graph, list(
            path.path), ax=ax, save=True, filepath=filepath, dpi=300, show=False)
        fig1.savefig(filepath, show=False, dpi=300, bbox_inches='tight')
        self.filenames.append(filepath)

    def fetch_frame_filepath(self) -> set:
        '''fetch next frame value from an infinite iterator that
        return consequitive numbers'''
        id = next(self.frame_num)
        return f'{dirs.GIFS}{id}.png'

    def clear_directory(self, directory='gifs/'):
        '''remove all contents of the directory'''
        if not os.path.exists(directory):
            print(f"The directory {directory} does not exist")
            return

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

# GifCreator().clear_directory()
