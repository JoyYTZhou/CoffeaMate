from dask.distributed import get_client
import psutil, sys, logging, gc, dask, os
from pympler import muppy, summary
from datetime import datetime
from typing import Any
from heapq import nlargest
from operator import attrgetter, itemgetter

SIZE_THRESHOLD = 10 * 1024 * 1024  # 10MB in bytes
MAX_OBJECTS = 10  # Maximum number of objects to log

def check_open_files() -> tuple[int, list[str]]:
    """Check and log details about currently open files with error handling.
    
    Returns:
        tuple: (number of open files, list of file paths)
    """
    try:
        process = psutil.Process()
        open_files = process.open_files()
        
        # Filter and sort by size if possible
        file_info = []
        for file in open_files:
            try:
                if os.path.exists(file.path):
                    size = os.path.getsize(file.path)
                    file_info.append((file, size))
            except (OSError, IOError) as e:
                logging.warning(f"Could not get size for {file.path}: {e}")
        
        # Sort by size if we have size info
        if file_info:
            file_info.sort(key=lambda x: x[1], reverse=True)
            largest_files = file_info[:MAX_OBJECTS]
        else:
            largest_files = [(f, 0) for f in open_files[:MAX_OBJECTS]]

        # Log summary
        logging.info(f"Total open files: {len(open_files)}")
        
        # Log details of largest/first 10 files
        for file, size in largest_files:
            try:
                file_details = {
                    'path': file.path,
                    'fd': file.fd,
                    'mode': file.mode if hasattr(file, 'mode') else 'unknown',
                    'size': f"{size/1024/1024:.2f}MB" if size > 0 else 'unknown',
                    'position': file.position if hasattr(file, 'position') else 'unknown'
                }
                logging.info(f"Open File Details: {file_details}")
            except Exception as e:
                logging.warning(f"Error getting details for file {file.path}: {e}")

        return len(open_files), [f[0].path for f in largest_files]

    except psutil.AccessDenied:
        logging.error("Access denied when trying to get open files")
        return 0, []
    except psutil.NoSuchProcess:
        logging.error("Process not found")
        return 0, []
    except Exception as e:
        logging.error(f"Unexpected error in check_open_files: {str(e)}")
        return 0, []

def analyze_memory_status(use_pympler=False):
    """Comprehensive memory analysis with optional Pympler details."""
    # Basic RSS memory report
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 3)
    logging.info(f"Current RSS memory usage: {memory_usage:.2f} GB")
    
    # System memory status
    system_memory = psutil.virtual_memory()
    if system_memory.used > SIZE_THRESHOLD:
        logging.warning(f"System memory usage: {system_memory.percent}% ({system_memory.used / 1e6:.2f} MB)")
    
    # Optional Pympler analysis
    if use_pympler:
        all_objects = muppy.get_objects()
        sum_obj = summary.summarize(all_objects)
        summary_str = "\n".join(summary.format_(sum_obj))
        logging.info(f"Pympler Memory Summary:\n{summary_str}")

def track_memory(track_objects=True, track_dask=True, interval=60, duration=600):
    """Combined memory tracking function."""
    import time
    start_time = time.time()
    end_time = start_time + duration
    
    while time.time() < end_time:
        if track_objects:
            analyze_memory()
        
        if track_dask:
            try:
                log_dask_status()
            except Exception as e:
                logging.error(f"Error tracking Dask status: {e}")
        
        time.sleep(interval)

def analyze_references(track_cycles=True, min_refs=10):
    """Combined reference analysis function."""
    gc.collect()
    large_objects = []
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

    # Analyze large objects and their references
    for obj in gc.get_objects():
        obj_size = get_object_size(obj)
        if obj_size > SIZE_THRESHOLD:
            referents = gc.get_referents(obj)
            if len(referents) > min_refs:
                large_objects.append((obj, obj_size, referents))
            if track_cycles:
                find_cycle(obj, [], set())

    # Log reference information
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

    # Log cycle information if tracked
    if track_cycles and cycles:
        largest_cycles = nlargest(MAX_OBJECTS, 
                                [(cycle, sum(get_object_size(obj) for obj in cycle)) 
                                 for cycle in cycles],
                                key=itemgetter(1))
        for cycle, total_size in largest_cycles:
            logging.warning(f"Found reference cycle (total size: {total_size/1e6:.2f}MB):")
            for obj in cycle:
                logging.warning(f"  {type(obj).__name__}: {get_object_size(obj)/1e6:.2f}MB")

def monitor_dask(track_growth=True):
    """Monitor Dask workers' memory usage and optionally track growth."""
    try:
        client = get_client()
        workers = client.scheduler_info()['workers']
        
        # Current memory status
        large_workers = []
        worker_history = {}
        
        for worker_id, worker in workers.items():
            if 'memory' in worker and worker['memory'] > SIZE_THRESHOLD:
                large_workers.append((worker_id, worker['memory']))
                worker_history[worker_id] = worker['memory']
        
        # Log current status
        largest_workers = nlargest(MAX_OBJECTS, large_workers, key=itemgetter(1))
        if largest_workers:
            logging.warning("Top Dask workers by memory usage:")
            for worker_id, memory in largest_workers:
                logging.warning(f"Worker {worker_id}: {memory / 1e6:.2f} MB")
        
        # Track growth if requested
        if track_growth:
            growing_workers = []
            current_workers = client.scheduler_info()['workers']
            
            for worker_id, worker in current_workers.items():
                if ('memory' in worker and worker['memory'] > SIZE_THRESHOLD and
                    worker_id in worker_history and 
                    worker['memory'] > worker_history[worker_id] * 1.5):
                    growing_workers.append((
                        worker_id,
                        worker_history[worker_id],
                        worker['memory']
                    ))
            
            if growing_workers:
                largest_growth = nlargest(MAX_OBJECTS, growing_workers, 
                                        key=lambda x: x[2] - x[1])
                logging.warning("Workers with significant memory growth:")
                for worker_id, old_mem, new_mem in largest_growth:
                    logging.warning(
                        f"Worker {worker_id}: {old_mem/1e6:.2f}MB → {new_mem/1e6:.2f}MB"
                    )
    except Exception as e:
        logging.error(f"Error monitoring Dask: {e}")  

def get_object_size(obj):
    """Safely get object size"""
    try:
        return sys.getsizeof(obj)
    except Exception:
        return 0

def analyze_memory_usage(include_types=True, include_collections=True):
    """Comprehensive memory analysis combining analyze_memory() and get_large_storage()."""
    gc.collect()
    objects = gc.get_objects()
    type_stats = {}
    large_collections = []

    # Analyze all objects
    for obj in gc.get_objects():
        try:
            obj_size = get_object_size(obj)
            if obj_size > SIZE_THRESHOLD:
                # Track by type
                if include_types:
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
                
                # Track collections
                if include_collections and isinstance(obj, (list, dict, tuple)):
                    large_collections.append((obj, obj_size))
        except Exception:
            logging.exception(f"Error processing object: {obj}")

    # Log type statistics
    if include_types:
        largest_types = nlargest(MAX_OBJECTS, type_stats.items(), 
                               key=lambda x: x[1]['size'])
        if largest_types:
            logging.info("Largest object types:")
            for type_name, stats in largest_types:
                size_mb = stats['size'] / (1024 * 1024)
                logging.info(f"{type_name}: Count={stats['count']}, Size={size_mb:.2f}MB")
                
                largest_examples = nlargest(3, stats['largest_objects'], 
                                          key=itemgetter(1))
                for obj, obj_size in largest_examples:
                    logging.debug(f"Example ({obj_size/1e6:.2f}MB): {str(obj)[:100]}")

    # Log collection details
    if include_collections and large_collections:
        largest_collections = nlargest(MAX_OBJECTS, large_collections, key=itemgetter(1))
        logging.debug("Largest collections:")
        for obj, size in largest_collections:
            logging.debug(f"Collection - Type: {type(obj)}, Size: {size/1e6:.2f} MB")
            length = len(obj) if isinstance(obj, (list, tuple)) else len(obj.keys())
            ele_print = min(5, length)
            if isinstance(obj, list):
                logging.debug(f"First {ele_print} elements: {obj[:ele_print]}")
            elif isinstance(obj, dict):
                logging.debug(f"First {ele_print} keys: {list(obj.keys())[:ele_print]}")
                logging.debug(f"First {ele_print} values: {list(obj.values())[:ele_print]}")
            elif isinstance(obj, tuple):
                logging.debug(f"First {ele_print} elements: {obj[:ele_print]}")

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
