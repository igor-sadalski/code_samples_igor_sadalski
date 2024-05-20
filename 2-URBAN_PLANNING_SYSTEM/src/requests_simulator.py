from datetime import datetime
from supporting_DS import DateTime, RequestChain, Request
import statistics
import osmnx as ox
from collections import Counter, deque
import pandas as pd
from typing import NamedTuple, Generator
from dataclasses import dataclass
import dirs
import config
from utilities import  log_runtime_and_memory


class RowHistoricData(NamedTuple):
    '''row in our dataset, immutable'''
    node_origin: int
    node_destination: int
    request_creation_time: DateTime
    requested_pickup_time: DateTime
    num_passengers: int  # TODO check this to correct unit types
    id: int  # TODO idealy hased value coming into the sytem, hashed based on all of the information provided, must be allocated new for incoming reuqeust
    booking_type: str  # TODO this must be also read by the system

#TODO setup the bus operation hours


class Data:
    '''load historical data from xlsx to memory (i.e. what we assume is know at for the algorithm)
    and simulator (i.e. unseen requests we will feed into the system during runtime)'''
    def __init__(self): #TODO automaticaly setup start and end date if they are not given
        self.df = self._create_dataframe_from_xlsx() #TODO remove nested __df imports, fix passing by refernce and class attibute
        self.initial_mem, self.initial_sim = self.initial_slice()
        self.memory = Memory(self.initial_mem)
        self.simulator = Simulator(self.initial_sim)
        # self._save_csvs() #TODO not working fix in the future

    def get_next(self) -> Request:
        '''retrive next request from the future; append it to memory; then perform 
        the typical rollout of the algorithm'''
        next_request = self.simulator.get_next()
        # self.memory. #automaticaly update memory in the future
        return next_request
    
    def simulation_iterator(self) -> Generator[Request, None, None]:
        '''iterate over simulator and return values from it '''
        while self.simulator.future_requests:
            yield self.get_next()
    
    def reset_simulator(self):
        '''return simulator the intial starting point'''
        self.simulator = Simulator(self.initial_sim)
    
    def are_any_requests(self)-> bool:
        '''check if there are any requests at all in the simulator;
        if simulator is exhausted stop the simulation; time independent
        just look at the queue'''
        return bool(self.simulator.future_requests) 

    def is_new_request(self, time: datetime) -> bool:
        '''check if there are any available requests up until and including
        the specified datetime; peek at the request creation time of the 
        newest request and add it if necessary'''
        return bool(self.simulator.future_requests[0].pickup_time <= time)

    # =========PRIVATE============
    def initial_slice(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''artificialy divide the dataset into requests we assume where known
        when the system started and these that were supposoed to income in the
        future; in the future you can also set and end date or optionaly use all
        requests; there will always be at least one such value!'''
        filtered_df = self.df[self.df['Request Creation Time'] >= config.HISTORY_END_TIME]
        split_index = filtered_df.index[0]
        initial_mem = self.df.iloc[:split_index]
        initial_sim = self.df.iloc[split_index:]
        return initial_mem, initial_sim

    @log_runtime_and_memory
    def _create_dataframe_from_xlsx(self) -> pd.DataFrame:
        '''convert xlsx data to private attibute; select approperiate time range;
        convert lat and lng to nodes and generate hash for each request (i.e. this is not
        strictly needed as request creation time should be already unique easier to work)'''
        columns_of_interest: list[str] = ["Request Creation Time", "Number of Passengers", "Booking Type",
                                          "Requested Pickup Time", "Origin Lat", "Origin Lng", "Destination Lat", "Destination Lng"]
        self.__df = pd.read_excel(
            dirs.HISTORIC_DATA, usecols=columns_of_interest)
        self.__df = self._select_range()
        self._convert_lat_lng_to_nodes()
        self._created_hashed_id()
        return self.__df #TODO remove the attibutes that are not longer used so they take less memroy

    def _save_csvs(self):
        self.df.to_csv(dirs.CSV_SAVE_REQUESTS_RANGE)
        self.initial_mem.to_csv(dirs.CSV_SAVE_REQUESTS_MEM)
        self.initial_sim.to_csv(dirs.CSV_SAVE_REQUESTS_SIM)

    def _created_hashed_id(self):
        '''create hash based on request creation time'''
        self.__df['Request ID'] = self.__df['Request Creation Time'].apply(hash)

    def _convert_lat_lng_to_nodes(self):
        '''find nearest nodes corresponding to request in the graph'''
        graph = ox.load_graphml(dirs.GRAPH_STRUCUTRE)
        self.__df["Origin Node"] = ox.nearest_nodes(
            graph, self.__df["Origin Lng"], self.__df["Origin Lat"])
        self.__df["Destination Node"] = ox.nearest_nodes(
            graph, self.__df["Destination Lng"], self.__df["Destination Lat"])
        self.__df.drop(["Origin Lat", 'Origin Lng', 'Destination Lat',
                       'Destination Lng'], axis=1, inplace=True)

    def _select_range(self):
        '''select only dates from the specified range'''  # TODO add support to check all or first n datapoints, log metadata for the selectd rangess
        return self.__df[(self.__df['Request Creation Time'] >= config.TIME_RANGE_START) & (self.__df['Request Creation Time'] <= config.TIME_RANGE_END)]

    

class Memory:  # TODO we must import two different files and combine them together
    '''download data and convert it dataclass; split the data into '''
    def __init__(self, dataframe: pd.DataFrame):
        self.historic_requests: list[RowHistoricData] = self._build(dataframe) #TODO add incoming requests as a different datatype

    def compute_mean_and_var(self) -> tuple[float, float]:
        '''based on whole data set estimate how many requests per day
        have been made, model this as normal distribution'''  # TODO this needs to be changed to the last few days
        #TODO change the datetime to actual timestamp models
        daily_requests = Counter([historic_request.requested_pickup_time.day
                                  for historic_request in self.historic_requests]).values()  # must use days before time we actulay test our model
        return statistics.mean(daily_requests), statistics.variance(daily_requests)
    
     # TODO add support to restrict the range when tou download the datetime
    @log_runtime_and_memory
    def _build(self, dataframe: pd.DataFrame) -> list[RowHistoricData]:
        '''convert df to our own datastructure'''
        rows = [RowHistoricData(row['Origin Node'], row['Destination Node'],
                                row['Request Creation Time'], row['Requested Pickup Time'],
                                row['Number of Passengers'], row['Request ID'], row['Booking Type']) 
                                for _, row in dataframe.iterrows()]
        return rows
    
class Simulator:
    '''class to yield future incoming requests'''
    def __init__(self, dataframe: pd.DataFrame):
        self.future_requests: deque[Request] = self._build(dataframe)

    def get_next(self) -> Request:
        '''get next stored future request from the deque; this automaticaly
        remove it from the stored and memorised requests in the simulator'''
        return self.future_requests.popleft() #TODO after popping from request we should automaticaly add to the historic data bank

    def append_new_request(self, new_request: Request):
        '''in case new requests come into the system append them to the deque of the system'''
        self.future_requests.append(new_request)


    @log_runtime_and_memory
    def _build(self, dataframe: pd.DataFrame) -> deque[Request]:
        '''convert df to our own datastructure'''
        requests = [Request(row['Origin Node'], row['Destination Node'], row['Requested Pickup Time'], 
                        row['Number of Passengers'], row['Request ID']) 
                                for _, row in dataframe.iterrows()]
        return deque(requests)
    
# dt = Data()
# mem = dt.memory
# initial_req = dt.get_next()