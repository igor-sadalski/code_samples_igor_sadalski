import os
import osmnx as ox
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import matplotlib.image as image
import matplotlib.pyplot as plt
import dirs

def create_video(policy_name='base_policy'):
    os.system('ffmpeg -r 1 -start_number 0 -i data/results/'+policy_name+'/frame%0d.png -pix_fmt yuvj420p -vcodec mjpeg -f mov data/results/'+policy_name+'/trajectory.mov')


class Plot_utils:
    def __init__(self, num_buses) -> None:
        self.bus_route_colors = self._generate_bus_route_colors()
        self.outstanding_requests_color = self._generate_outstanding_requests_color()
        self.frame_number = 0
        self.num_buses = num_buses
    
    def reset_frame_number(self):
        self.frame_number = 0
    
    def _generate_bus_route_colors(self):
        bus_colors = ["#61cbf4"] #8ed973", "#d86ecc"
        return bus_colors
    
    def _generate_outstanding_requests_color(self):
        request_color = "#ff8585"
        return request_color
    
    def _load_bus_icons(self):
        blue_bus = image.imread(dirs.BUS_ICON)
        bus_icons = [blue_bus]

        blue_flag = image.imread(dirs.FLAG_ICON)
        flag_icons = [blue_flag]

        red_marker = image.imread(dirs.MARKER_ICON)
        red_passenger = image.imread(dirs.PASSENGER_ICON)

        return bus_icons, flag_icons, red_marker, red_passenger
    
    def _add_anotation_to_map(self, axis, icon, node_key, zoom=0.2):
        node = config.MAP.graph.nodes[node_key]
        node_latitude = node['y']
        node_longitude = node['x']
        imagebox = OffsetImage(icon, zoom=zoom)
        ab = AnnotationBbox(imagebox, (node_longitude, node_latitude), frameon = False)
        axis.add_artist(ab)

    def _insert_outstanding_request_annotations(self, map_object, route_colors, routes_to_plot, ax, red_passenger, 
                                                red_marker, origin, destination, zoom=0.2):
        # Add annotation for the origin of the outstanding request
        self._add_anotation_to_map(map_object=map_object, axis=ax, icon=red_passenger, node_key=origin, zoom=zoom)

        # Add annotation for the destination of the outstanding request
        self._add_anotation_to_map(map_object=map_object, axis=ax, icon=red_marker, node_key=destination, zoom=zoom)

import config
pic = Plot_utils(1)
bus_icons, flag_icons, red_marker, red_passenger = pic._load_bus_icons()
fig, ax = ox.plot_graph(config.MAP.graph, node_size=1, edge_linewidth=0.5, node_alpha=0.8, show=False) #if needed  node_color=map_object.colors_w
# pic._add_anotation_to_map(ax, dirs.BUS_ICON, 61328936, zoom=0.2)

    
    def _insert_annotations_for_buses_and_stops(self, map_object, ax, flag_icons, bus_icons, bus_locations, bus_stops, zoom=0.2):
        for i in range(self.num_buses):
            bus_index = i % 3
            bus_icon = bus_icons[bus_index]
            location = bus_locations[bus_index]

            self._add_anotation_to_map(map_object=map_object, axis=ax, icon=bus_icon, node_key=location, zoom=zoom)
        
        for j, stops in enumerate(bus_stops):
            bus_index = j % 3
            flag_icon = flag_icons[bus_index]

            for index, bus_stop in enumerate(stops):
                if index != 0 and index != len(stops)-1:
                    self._add_anotation_to_map(map_object=map_object, axis=ax, icon=flag_icon, node_key=bus_stop, zoom=zoom)
    
    def _plot_bus_routes(self, map_object, ax, bus_routes, route_colors, routes_to_plot, filepath):
        for k, route in enumerate(bus_routes):
            bus_index = k % 3
            route_color = self.bus_route_colors[bus_index]
            if len(route) > 2:
                route_colors.append(route_color)
                routes_to_plot.append(route)
        if len(routes_to_plot) == 1:
            ox.plot_graph_route(map_object.G, ax=ax, filepath=filepath, save=True, route=routes_to_plot[0], route_color=route_colors[0], 
                                route_linewidth=3, node_size=1, show=False, close=True)
        elif len(routes_to_plot) > 1:
            ox.plot_graph_routes(map_object.G, ax=ax, filepath=filepath, save=True, routes=routes_to_plot, route_colors=route_colors, 
                                route_linewidths=3, node_size=1, show=False, close=True)
        else:
            ox.plot_graph(map_object.G, ax=ax, filepath=filepath, save=True, node_size=1, edge_linewidth=0.5, node_alpha=0.8, show=False, close=True)

    
    def plot_routes_before_assignment_offline(self, map_object, current_assignment, request_assignment, prev_bus_stops, prev_bus_routes, bus_locations, 
                                      folder_path="../results/trajectories"):
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        bus_icons, flag_icons, red_marker, red_passenger = self._load_bus_icons()
        filepath = os.path.join( folder_path, "frame"+str(self.frame_number)+".png")
        fig, ax = ox.plot_graph(map_object.G, node_color=map_object.colors_w, node_size=1, edge_linewidth=0.5, node_alpha=0.8, show=False)

        route_colors = []
        routes_to_plot = []
        for bus_index in range(self.num_buses):
            current_request_index_list = current_assignment[bus_index]
            for request_index in current_request_index_list:
                current_request_row = request_assignment[bus_index][request_index]
                request_origin = current_request_row["Origin Node"]
                request_destination = current_request_row["Destination Node"]
                self._insert_outstanding_request_annotations(map_object=map_object, route_colors=route_colors, routes_to_plot=routes_to_plot, 
                                                            ax=ax, red_passenger=red_passenger, red_marker=red_marker, origin=request_origin, 
                                                            destination=request_destination)
        
        self._insert_annotations_for_buses_and_stops(map_object=map_object, ax=ax, flag_icons=flag_icons, bus_icons=bus_icons, 
                                                     bus_locations=bus_locations, bus_stops=prev_bus_stops)
        
        
        self._plot_bus_routes(map_object=map_object, ax=ax, bus_routes=prev_bus_routes, route_colors=route_colors, routes_to_plot=routes_to_plot,
                              filepath=filepath)
        
        plt.close()

        self.frame_number += 1


self

def preprocessed_data