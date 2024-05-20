import os
#jupyter
# get_absolute_path = lambda relative_path: os.path.join(os.getcwd(), '../', relative_path)
def get_absolute_path(relative_path): return os.path.join(os.getcwd(), relative_path)

GRAPH_STRUCUTRE = get_absolute_path('data/graph_data/graph_structure.graphml')
HISTORIC_DATA = get_absolute_path('data/requests/HTS-requests_2022.xlsx')

MAP_SHORTESTS_PATH = get_absolute_path('data/graph_data/path_dict.pkl')
MAP_TIME = get_absolute_path('data/graph_data/path_time_dict.pkl')

BUS_ICON = get_absolute_path('bus_icons/blue_bus.png')
FLAG_ICON = get_absolute_path('bus_icons/blue_flag.png')
MARKER_ICON = get_absolute_path('bus_icons/red_marker.png')
PASSENGER_ICON = get_absolute_path('bus_icons/red_passenger.png')

CSV_SAVE_REQUESTS_RANGE = get_absolute_path('data/requests/pre_processes_requests/selected_range_of_requests.csv')
CSV_SAVE_REQUESTS_MEM = get_absolute_path('data/requests/pre_processes_requests/initial_mem.csv')
CSV_SAVE_REQUESTS_SIM = get_absolute_path('data/requests/pre_processes_requests/initial_req.csv')

GIFS= 'gifs/plot_'