from dask.distributed import get_client
import psutil, sys, logging, gc, dask, os, ctypes, platform
from pympler import muppy, summary
from datetime import datetime
from typing import Any
from heapq import nlargest
from operator import attrgetter, itemgetter

SIZE_THRESHOLD = 10 * 1024 * 1024  # 10MB in bytes
MAX_OBJECTS = 10  # Maximum number of objects to log

def restart_if_vms_high(threshold_gb=6):
    """Restart Python if VMS memory usage is too high."""
    process = psutil.Process(os.getpid())
    vms = process.memory_info().vms / (1024**3)  # Convert to GB

    if vms > threshold_gb:
        logging.warning(f"🔴 High VMS detected ({vms:.2f} GB). Restarting Python...")
        os.execv(sys.executable, [sys.executable] + sys.argv)  # Restart process

def analyze_memory_status(process=None, use_pympler=False):
    """Comprehensive memory analysis with optional Pympler details."""
    if process is None:
        process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 3)
    logging.info(f"Current RSS memory usage: {memory_usage:.2f} GB")
    vms_usage = process.memory_info().vms / (1024 ** 3)
    logging.info(f"Current VMS memory usage: {vms_usage:.2f} GB")

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

def force_mallopt_trim():
    """Use mallopt() to force aggressive memory release in glibc."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        M_MMAP_THRESHOLD = 16  # Reduce mmap threshold
        M_TRIM_THRESHOLD = -1  # Force trimming
        libc.mallopt(0, M_MMAP_THRESHOLD)
        libc.mallopt(1, M_TRIM_THRESHOLD)
        logging.info("✅ Successfully called mallopt() to reduce memory fragmentation.")
    except Exception as e:
        logging.exception(f"⚠️ Error calling mallopt(): {e}")

def is_jemalloc_used():
    """Check if jemalloc is being used as the memory allocator."""
    try:
        with open("/proc/self/maps", "r") as f:
            for line in f:
                if "jemalloc" in line:
                    return True
    except Exception:
        logging.exception("Error checking memory allocator")
    return False

def force_release_memory():
    """Forces memory release by running garbage collection and `malloc_trim(0)` (if supported)."""
    try:
        gc.collect()
        if "glibc" in platform.libc_ver()[0] and not is_jemalloc_used():
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
            logging.info("Successfully released memory using malloc_trim(0).")
        else:
            logging.warning("Skipping malloc_trim(0): System is not using glibc or is using jemalloc.")
    except Exception as e:
        logging.exception(f"Error releasing memory: {e}")

def check_and_release_memory(process: psutil.Process, 
                           rss_threshold_gb: float = 4.0,
                           vms_rss_diff_threshold_mb: float = 400.0) -> None:
    """Check memory usage and force release if exceeding thresholds.
    
    Args:
        process: psutil.Process object to monitor
        rss_threshold_gb: Threshold for RSS memory in GB
        vms_rss_diff_threshold_mb: Threshold for VMS-RSS difference in MB
    """
    mem_info = process.memory_info()
    rss_gb = mem_info.rss / (1024 ** 3)  # Convert to GB
    vms_mb = mem_info.vms / (1024 ** 2)  # Convert to MB
    rss_mb = mem_info.rss / (1024 ** 2)  # Convert to MB
    vms_rss_diff = vms_mb - rss_mb
    
    if rss_gb > rss_threshold_gb and vms_rss_diff > vms_rss_diff_threshold_mb:
        logging.warning(
            f"High memory usage detected:\n"
            f"RSS: {rss_gb:.2f}GB (threshold: {rss_threshold_gb}GB)\n"
            f"VMS-RSS diff: {vms_rss_diff:.2f}MB (threshold: {vms_rss_diff_threshold_mb}MB)"
        )
        logging.info("Forcing memory release...")
        force_release_memory()
        
        # Log memory after release
        new_mem = process.memory_info()
        new_rss_gb = new_mem.rss / (1024 ** 3)
        new_vms_mb = new_mem.vms / (1024 ** 2)
        new_vms_rss_diff = new_vms_mb - (new_mem.rss / (1024 ** 2))
        logging.info(f"Memory after forced release:\n"
                      f"RSS: {new_rss_gb:.2f}GB\n"
                      f"VMS: {new_vms_mb:.2f}GB\n"
                      f"VMS-RSS diff: {new_vms_rss_diff:.2f}MB")
        if new_rss_gb > rss_threshold_gb and new_vms_rss_diff > vms_rss_diff_threshold_mb:
            logging.warning("Memory release did not reduce memory usage sufficiently.")
            logging.warning("Force mallopt trim to reduce memory fragmentation.")
            force_mallopt_trim()
            analyze_memory_status(process, use_pympler=False)

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

def track_memory_all(track_objects=True, track_dask=True, interval=60, duration=600):
    """Combined memory tracking function.
    
    Args:
        track_objects (bool): Whether to track object memory usage
        track_dask (bool): Whether to track Dask worker memory
        interval (int): Seconds between checks
        duration (int): Total tracking duration in seconds
    """
    import time
    start_time = time.time()
    end_time = start_time + duration
    process = psutil.Process(os.getpid())
    
    while time.time() < end_time:
        if track_objects:
            # Use the new consolidated function instead of analyze_memory
            analyze_memory_usage(include_types=True, include_collections=True)
            # Also get basic memory status
            analyze_memory_status(process, use_pympler=False)
        
        if track_dask:
            try:
                # Use the new consolidated Dask monitoring function
                monitor_dask(track_growth=True)
            except Exception as e:
                logging.error(f"Error tracking Dask status: {e}")
        
        # Log current process memory
        process = psutil.Process()
        log_memory(process, f"Checkpoint at {time.strftime('%H:%M:%S')}")
        
        time.sleep(interval)

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

def get_object_size(obj):
    """Safely get object size"""
    try:
        return sys.getsizeof(obj)
    except Exception:
        return 0

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