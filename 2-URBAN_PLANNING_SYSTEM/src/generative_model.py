from collections import Counter
import random
import statistics
import config
from supporting_DS import Request, RequestChain, Date, Time, DateTime
import dirs
from utilities import  log_runtime_and_memory
from requests_simulator import *

# TODO change generative models to the approperiate classes and dataclasses

        
class RequestsHistogram(NamedTuple):
    '''frequency of requests in the historic data set'''
    requests: RequestChain
    weights: list[int]

    def sample(self, request_number: int) -> RequestChain:
        '''sample a chain of requests of length n_chains 
        from from the histogram of the historic data requests'''
        sampled_chain = random.choices(self.requests.requests_chain,
                                       weights=self.weights,
                                       k=request_number)
        return RequestChain(requests_chain=sampled_chain)


class RequestBank(NamedTuple):
    '''bank or sampled request chains'''
    values: list[RequestChain]


class GenerativeModel:
    '''build offline bank of bootstrapped requests from dataset;
    each request chain should have the length estimated by the normal
    distribution computed based on the available dataset'''
    #TODO mark change private attibutes of the functions with the underscore 
    def __init__(self, historic_data):
        self._historic_data: Memory = historic_data
        self._preprocessed_historic_data: RequestChain = self._preprocess_data()
        # TODO add specific type for list of RequestChain and RowHistoricDAta
        self._requests_bank: RequestBank = self._build_bank()

    # =========PUBLIC============

    def sample_request_from_the_bank(self) -> RequestChain:
        '''public method used to online sample requests from offline
        computed bank of chains of requests'''
        return random.choice(self._requests_bank.values)
    
    # =========PRIVATE============

    # TODO no cheating with how we desing our system
    def _preprocess_data(self) -> RequestChain:
        '''delete date from datetime in historic data, copy the rest'''
        table = [Request(row.node_origin, row.node_destination, row.requested_pickup_time.time, 
                         row.num_passengers, row.id) for row in self._historic_data.historic_requests]
        return RequestChain(table)

    def _create_histogram(self) -> RequestsHistogram:
        '''compute the frequency for each requests in the historic 
        data set'''
        counter = Counter(self._preprocessed_historic_data.requests_chain)
        requests = RequestChain(list(counter.keys()))
        weights = list(counter.values())
        return RequestsHistogram(requests, weights)

    @log_runtime_and_memory
    def _build_bank(self) -> RequestBank:
        '''boostrap from histogram data many requests,
        computed offline sampled online during runtime'''
        mu, var = self._historic_data.compute_mean_and_var()  # TODO fix
        # TODO must use days before time we actulay test our model
        self.histogram: RequestsHistogram = self._create_histogram()
        num_requests = int(random.gauss(mu, var**0.5))
        values = [self.histogram.sample(num_requests)
                            for _ in range(config.SAMPLED_BANK_SIZE)]
        return RequestBank(values)  # will this reevalue each time at runtime

    #=======UTILITES=====

    @property
    def historic_data(self):
        '''getter for historic data; allow to read private class attribute'''
        return self._historic_data

    @historic_data.setter
    def historic_data(self, value):
        '''setter for historic data; must rebuild bank each time new request is made
        take care of this automaticaly''' 
        self._historic_data = value
        self._preprocessed_historic_data = self._preprocess_data()
        self._requests_bank = self._build_bank()

# dt = Data()
# mem = dt.memory
# gen = GenerativeModel(dts.memory)
# out = gen.sample_request_from_the_bank()
