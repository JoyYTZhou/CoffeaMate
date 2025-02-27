from dask.distributed import get_client
import psutil, sys, logging, gc, dask
from datetime import datetime
from typing import Any
from heapq import nlargest
from operator import attrgetter, itemgetter

# Common thresholds
SIZE_THRESHOLD = 10 * 1024 * 1024  # 10MB in bytes
MAX_OBJECTS = 10  # Maximum number of objects to log

def track_object_growth(interval=60, duration=600):
    """Track object count growth over time to identify potential leaks.
    
    Args:
        interval (int): Seconds between checks
        duration (int): Total tracking duration in seconds
    """
    import time
    type_counts = {}
    iterations = duration // interval
    
    for i in range(iterations):
        gc.collect()
        current_counts = {}
        
        # Count objects larger than threshold
        for obj in gc.get_objects():
            if get_object_size(obj) > SIZE_THRESHOLD:
                obj_type = type(obj).__name__
                current_counts[obj_type] = current_counts.get(obj_type, 0) + 1
        
        # Compare with previous counts
        for obj_type, count in current_counts.items():
            if obj_type in type_counts:
                if count > type_counts[obj_type]:
                    logging.warning(f"Potential leak: {obj_type} grew from {type_counts[obj_type]} to {count}")
            type_counts[obj_type] = count
            
        time.sleep(interval)

def find_reference_cycles():
    """Find reference cycles among large objects."""
    gc.collect()
    cycles = []
    
    def find_cycle(obj, path, visited):
        if id(obj) in visited:
            cycle_start = path.index(obj)
            cycles.append(path[cycle_start:])
            return
        visited.add(id(obj))
        path.append(obj)
        
        for ref in gc.get_referents(obj):
            if get_object_size(ref) > SIZE_THRESHOLD:
                find_cycle(ref, path[:], visited)
    
    for obj in gc.get_objects():
        if get_object_size(obj) > SIZE_THRESHOLD:
            find_cycle(obj, [], set())
    
    # Log the largest cycles
    largest_cycles = nlargest(MAX_OBJECTS, 
                            [(cycle, sum(get_object_size(obj) for obj in cycle)) 
                             for cycle in cycles],
                            key=itemgetter(1))
    
    for cycle, total_size in largest_cycles:
        logging.warning(f"Found reference cycle (total size: {total_size/1e6:.2f}MB):")
        for obj in cycle:
            logging.warning(f"  {type(obj).__name__}: {get_object_size(obj)/1e6:.2f}MB")

def monitor_generation_counts():
    """Monitor objects in different garbage collection generations."""
    gc.collect()  # Clean up first
    
    # Get counts before and after collection
    counts_before = gc.get_count()
    gc.collect()
    counts_after = gc.get_count()
    
    for gen in range(3):
        if counts_before[gen] - counts_after[gen] > 100:  # Significant cleanup
            logging.warning(f"Generation {gen} cleaned up {counts_before[gen] - counts_after[gen]} objects")
            
        # Check surviving large objects in each generation
        survivors = []
        for obj in gc.get_objects():
            if get_object_size(obj) > SIZE_THRESHOLD:
                survivors.append((obj, get_object_size(obj)))
        
        largest_survivors = nlargest(MAX_OBJECTS, survivors, key=itemgetter(1))
        if largest_survivors:
            logging.warning(f"Largest surviving objects in generation {gen}:")
            for obj, size in largest_survivors:
                logging.warning(f"  {type(obj).__name__}: {size/1e6:.2f}MB")

def track_dask_leaks():
    """Track potential memory leaks in Dask operations."""
    client = get_client()
    initial_workers = client.scheduler_info()['workers']
    
    # Track worker memory over time
    worker_history = {}
    
    for worker_id, worker in initial_workers.items():
        if 'memory' in worker and worker['memory'] > SIZE_THRESHOLD:
            worker_history[worker_id] = worker['memory']
    
    # Check for consistently growing workers
    current_workers = client.scheduler_info()['workers']
    growing_workers = []
    
    for worker_id, worker in current_workers.items():
        if 'memory' in worker and worker['memory'] > SIZE_THRESHOLD:
            if (worker_id in worker_history and 
                worker['memory'] > worker_history[worker_id] * 1.5):  # 50% growth
                growing_workers.append((
                    worker_id,
                    worker_history[worker_id],
                    worker['memory']
                ))
    
    # Log the top growing workers
    largest_growth = nlargest(MAX_OBJECTS, growing_workers, 
                            key=lambda x: x[2] - x[1])
    
    for worker_id, old_mem, new_mem in largest_growth:
        logging.warning(
            f"Worker {worker_id} memory grew from "
            f"{old_mem/1e6:.2f}MB to {new_mem/1e6:.2f}MB"
        )
    
    
def get_object_size(obj):
    """Safely get object size"""
    try:
        return sys.getsizeof(obj)
    except Exception:
        return 0

def get_large_storage():
    """Log details of top 10 largest collections (>10MB)"""
    large_objects = []
    
    for obj in gc.get_objects():
        if isinstance(obj, (list, dict, tuple)):
            obj_size = get_object_size(obj)
            if obj_size > SIZE_THRESHOLD:
                large_objects.append((obj, obj_size))
    
    # Get top 10 largest objects
    largest_objects = nlargest(MAX_OBJECTS, large_objects, key=itemgetter(1))
    
    for obj, size in largest_objects:
        logging.debug(f"Large object - Type: {type(obj)}, Size: {size / 1e6:.2f} MB")
        length = len(obj) if isinstance(obj, (list, tuple)) else len(obj.keys())
        ele_print = min(5, length)
        if isinstance(obj, list):
            logging.debug(f"First {ele_print} elements: {obj[:ele_print]}")
        elif isinstance(obj, dict):
            logging.debug(f"First {ele_print} keys: {list(obj.keys())[:ele_print]}")
            logging.debug(f"First {ele_print} values: {list(obj.values())[:ele_print]}")
        elif isinstance(obj, tuple):
            logging.debug(f"First {ele_print} elements: {obj[:ele_print]}")

def get_reference(num_refs=10):
    """Log top 10 largest objects (>10MB) with many referents"""
    large_objects = []
    
    for obj in gc.get_objects():
        obj_size = get_object_size(obj)
        if obj_size > SIZE_THRESHOLD:
            referents = gc.get_referents(obj)
            if len(referents) > num_refs:
                large_objects.append((obj, obj_size, referents))
    
    # Get top 10 largest objects
    largest_objects = nlargest(MAX_OBJECTS, large_objects, key=itemgetter(1))
    
    for obj, size, referents in largest_objects:
        if hasattr(dask, "delayed") and isinstance(obj, dask.delayed.Delayed):
            logging.debug(f"Large delayed object: {obj} ({size / 1e6:.2f} MB)")
        elif type(obj).__module__.endswith('analysis.processor'):
            logging.debug(f"Large processor object - Type: {type(obj)}, Size: {size / 1e6:.2f} MB")
            referrers = gc.get_referrers(obj)
            logging.debug(f"Referrers: {referrers}")
        else:
            logging.debug(f"Large object {obj} ({size / 1e6:.2f} MB) has {len(referents)} referents")

def analyze_memory():
    """Analyze top 10 largest memory users by type (>10MB)"""
    gc.collect()
    objects = gc.get_objects()
    type_stats = {}

    # Group objects by type
    for obj in objects:
        try:
            obj_size = get_object_size(obj)
            if obj_size > SIZE_THRESHOLD:
                obj_type = type(obj).__name__
                if obj_type not in type_stats:
                    type_stats[obj_type] = {
                        'count': 0,
                        'size': 0,
                        'largest_objects': []
                    }
                type_stats[obj_type]['count'] += 1
                type_stats[obj_type]['size'] += obj_size
                type_stats[obj_type]['largest_objects'].append((obj, obj_size))
        except Exception:
            logging.exception(f"Error processing object: {obj}")
    
    # Get top 10 types by total size
    largest_types = nlargest(MAX_OBJECTS, type_stats.items(), 
                           key=lambda x: x[1]['size'])
    
    logging.info("Top 10 largest object types (>10MB):")
    for type_name, stats in largest_types:
        size_mb = stats['size'] / (1024 * 1024)
        logging.info(f"{type_name}: Count={stats['count']}, Size={size_mb:.2f}MB")
        
        # Get top 3 largest objects of this type
        largest_examples = nlargest(3, stats['largest_objects'], 
                                  key=itemgetter(1))
        for obj, obj_size in largest_examples:
            logging.debug(f"Example ({obj_size / 1e6:.2f}MB): {str(obj)[:100]}")

def log_dask_status():
    """Log top 10 largest Dask worker memory usage (>10MB)"""
    client = get_client()
    workers = client.scheduler_info()['workers']

    large_workers = []
    for worker_id, worker in workers.items():
        try:
            if 'memory' in worker and worker['memory'] > SIZE_THRESHOLD:
                large_workers.append((worker_id, worker['memory']))
        except Exception:
            continue

    # Get top 10 workers by memory usage
    largest_workers = nlargest(MAX_OBJECTS, large_workers, key=itemgetter(1))
    
    if largest_workers:
        logging.warning("Top 10 Dask workers with largest memory:")
        for worker_id, memory in largest_workers:
            logging.warning(f"Worker {worker_id}: {memory / 1e6:.2f} MB")

    system_memory = psutil.virtual_memory()
    if system_memory.used > SIZE_THRESHOLD:
        logging.warning(f"System memory usage: {system_memory.percent}% ({system_memory.used / 1e6:.2f} MB)")

def log_memory_snapshot(snapshot, message):
    """Log top 10 largest memory statistics (>10MB)"""
    stats = [stat for stat in snapshot.statistics('lineno') 
             if stat.size > SIZE_THRESHOLD]
    largest_stats = nlargest(MAX_OBJECTS, stats, key=attrgetter('size'))
    
    if largest_stats:
        logging.debug(f"Top 10 largest memory objects - {message}")
        for stat in largest_stats:
            logging.debug(f"{stat.size / 1e6:.2f}MB: {stat}")

def log_memory_diff(snapshot1, snapshot2, message):
    """Log top 10 largest memory differences (>10MB)"""
    stats = [stat for stat in snapshot2.compare_to(snapshot1, 'lineno') 
             if abs(stat.size_diff) > SIZE_THRESHOLD]
    largest_diffs = nlargest(MAX_OBJECTS, stats, 
                           key=lambda x: abs(x.size_diff))
    
    if largest_diffs:
        logging.debug(f"Top 10 largest memory differences - {message}")
        for stat in largest_diffs:
            logging.debug(f"{abs(stat.size_diff) / 1e6:.2f}MB: {stat}")

def log_memory(process, stage):
    """Log memory usage if large enough"""
    mem_usage = process.memory_info().rss
    if mem_usage > SIZE_THRESHOLD:
        logging.warning(f"Memory usage at {stage}: {mem_usage / 1e6:.2f} MB")
    return mem_usage

def setup_logging(console_level=logging.WARNING, file_level=logging.DEBUG):
    """Setup logging with consistent format and levels"""
    logging.getLogger().handlers.clear()
    debug_filename = f'debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        filename=debug_filename,
        level=file_level,
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    logging.getLogger().addHandler(console_handler)
    
    # Set specific logger levels
    logging.getLogger("uproot").setLevel(logging.WARNING)
    logging.getLogger("dask").setLevel(logging.DEBUG)
    logging.getLogger("distributed").setLevel(logging.DEBUG)
    logging.getLogger("fsspec").setLevel(logging.WARNING)
