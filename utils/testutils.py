from dask.distributed import get_client
import psutil, sys, logging, gc, dask
from datetime import datetime
from typing import Any

def get_reference(num_refs=10):
    """Log objects with more than 10 referents"""
    for obj in gc.get_objects():
        referents = gc.get_referents(obj)
        if len(referents) > num_refs:
            logging.warning(f"Object {obj} has {len(referents)} referents: {referents}")
        elif isinstance(obj, dask.delayed.Delayed):
            logging.warning(f"Delayed object: {obj}")
        elif type(obj).__module__.endswith('analysis.processor'):
            logging.warning(f"Type: {type(obj)}, Size: {sys.getsizeof(obj)} bytes")
            referrers = gc.get_referrers(obj)
            logging.debug(f"Referrers: {referrers}")

def analyze_memory():
    """Analyze memory usage by type"""
    gc.collect()  # Force garbage collection
    objects = gc.get_objects()
    stats = {}
    
    # Group objects by type
    for obj in objects:
        obj_type = type(obj).__name__
        if obj_type not in stats:
            stats[obj_type] = {
                'count': 0,
                'size': 0,
                'examples': []
            }
        stats[obj_type]['count'] += 1
        try:
            stats[obj_type]['size'] += get_size(obj)
            if len(stats[obj_type]['examples']) < 3:  # Keep up to 3 examples
                stats[obj_type]['examples'].append(str(obj)[:100])  # Truncate long strings
        except Exception as e:
            logging.debug(f"Error measuring size of {obj_type}: {e}")
    
    # Sort by size and print
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['size'], reverse=True)
    logging.info("Memory usage by type:")
    for type_name, type_stats in sorted_stats[:20]:  # Show top 20
        size_mb = type_stats['size'] / (1024 * 1024)
        if size_mb > 1:  # Only show objects using more than 1MB
            logging.info(f"{type_name}: Count={type_stats['count']}, Size={size_mb:.2f}MB")
            if type_stats['examples']:
                logging.debug(f"Examples: {type_stats['examples']}")

def get_size(obj: Any) -> int:
    """Get size of object and its members in bytes"""
    memory_size = 0
    processed_ids = set()

    def inner_size(obj: Any) -> int:
        obj_id = id(obj)
        if obj_id in processed_ids:
            return 0
        processed_ids.add(obj_id)
        
        size = sys.getsizeof(obj)
        
        if isinstance(obj, (tuple, list, set, dict)):
            size += sum(inner_size(item) for item in obj)
        elif hasattr(obj, '__dict__'):
            size += inner_size(obj.__dict__)
        elif isinstance(obj, dict):
            size += sum(inner_size(k) + inner_size(v) for k, v in obj.items())
            
        return size
    
    return inner_size(obj)

def log_memory(process, stage):
    """Logs memory usage at different stages using a provided psutil.Process() object."""
    mem_usage = process.memory_info().rss / (1024 * 1024)
    logging.warning(f"Memory usage at {stage}: {mem_usage:.2f} MB")
    return mem_usage

def log_dask_status():
    client = get_client()
    workers = client.scheduler_info()['workers']

    worker_memory = []
    for worker in workers.values():
        try:
            if 'memory' in worker:
                worker_memory.append(worker['memory'])
        except Exception as e:
            logging.warning(f"Could not get memory info from worker: {e}")

    if worker_memory:
        logging.warning(f"Dask workers memory: {worker_memory}")
    else:
        logging.warning("No memory information available from Dask workers")

    logging.warning(f"System memory: {psutil.virtual_memory().percent}%")

def setup_logging(console_level=logging.WARNING, file_level=logging.DEBUG):
    # Enhanced logging format for debugging
    logging.getLogger().handlers.clear()
    debug_filename = f'debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        filename='debug.log',
        level=file_level,
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    # Also show logs in console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    logging.getLogger().addHandler(console_handler)
    logging.getLogger("uproot").setLevel(logging.WARNING)
    logging.getLogger("dask").setLevel(logging.DEBUG)
    logging.getLogger("distributed").setLevel(logging.DEBUG)
    logging.getLogger("fsspec").setLevel(logging.WARNING)

def log_memory_snapshot(snapshot, message):
    top_stats = snapshot.statistics('lineno')
    logging.debug(f"Memory snapshot: {message}")
    for stat in top_stats[:10]:
        logging.debug(stat)
    
def log_memory_diff(snapshot1, snapshot2, message):
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    logging.debug(f"[ Memory differences after {message} ]")
    for stat in top_stats[:10]:
        logging.debug(stat)