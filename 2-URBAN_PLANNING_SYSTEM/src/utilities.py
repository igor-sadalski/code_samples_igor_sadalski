#TODO try to implement some of the https://www.brendangregg.com/flamegraphs.html
#flame graph for memory and the runtime complexity
import time
import logging
from memory_profiler import memory_usage
import functools

#enable this line if you want to clean old logs
with open('logfile.log', 'w'):
    pass

logging.basicConfig(filename='logfile.log', level=logging.INFO)

def log_runtime_and_memory(func):
    '''python decorator to log into external file function runtime and 
    memory usage'''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        mem_before = memory_usage(-1, interval=0.1, timeout=1)[0]

        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        mem_after = memory_usage(-1, interval=0.1, timeout=1)[0]

        if args and hasattr(args[0], '__class__'):
            logging.info(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Function {func.__name__} of class {args[0].__class__.__name__} took {end_time - start_time} seconds to run.')
            logging.info(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Function {func.__name__} of class {args[0].__class__.__name__} used {mem_after - mem_before} MiB.')
        else:
            logging.info(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Function {func.__name__} took {end_time - start_time} seconds to run.')
            logging.info(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Function {func.__name__} used {mem_after - mem_before} MiB.')

        return result
    return wrapper