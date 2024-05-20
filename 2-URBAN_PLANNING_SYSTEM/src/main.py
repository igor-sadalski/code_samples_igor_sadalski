'''set of functions to interface with simulator'''
from supporting_DS import BusRoute, BusID
from requests_simulator import *
# from MCTS import *

#copy example memory and incoming request to test the system


def main():
    '''compute the cost of algorithm of runing a planning sequence in the simulator
    with historical data and new requests coming in real time'''
    dt = Data()
    first_req = dt.get_next()     
    bus_route = BusRoute(BusID(0), planned_requests=[])
    bus_route.allocate(first_req)
    a  = 10
    # for cur_request in yield_new_request_simulator():
    #     mc_forest = MCForest(cur_paths, cur_request, historic_data)
    #     cur_paths = mc_forest.get_best_action()
    #     send_to_simulator(cur_paths)
    # return cost_of_running
    
main()