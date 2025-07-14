"""
Metrics and Monitoring for AML Engine
Prometheus metrics collection and monitoring
"""

from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.core import CollectorRegistry
import time
from functools import wraps
from typing import Dict, Any
import psutil
import os

# Create a custom registry
registry = CollectorRegistry()

# API Metrics
REQUEST_COUNT = Counter(
    'aml_api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

REQUEST_DURATION = Histogram(
    'aml_api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    registry=registry
)

ACTIVE_REQUESTS = Gauge(
    'aml_api_active_requests',
    'Number of active API requests',
    registry=registry
)

# Model Metrics
MODEL_PREDICTIONS = Counter(
    'aml_model_predictions_total',
    'Total number of model predictions',
    ['model_type', 'framework'],
    registry=registry
)

PREDICTION_DURATION = Histogram(
    'aml_model_prediction_duration_seconds',
    'Model prediction duration in seconds',
    ['model_type'],
    registry=registry
)

MODEL_ACCURACY = Gauge(
    'aml_model_accuracy',
    'Model accuracy score',
    ['model_type'],
    registry=registry
)

# Cache Metrics
CACHE_HITS = Counter(
    'aml_cache_hits_total',
    'Total number of cache hits',
    ['cache_type'],
    registry=registry
)

CACHE_MISSES = Counter(
    'aml_cache_misses_total',
    'Total number of cache misses',
    ['cache_type'],
    registry=registry
)

CACHE_SIZE = Gauge(
    'aml_cache_size',
    'Current cache size',
    ['cache_type'],
    registry=registry
)

# Alert Metrics
ALERTS_GENERATED = Counter(
    'aml_alerts_generated_total',
    'Total number of alerts generated',
    ['severity', 'type'],
    registry=registry
)

ACTIVE_ALERTS = Gauge(
    'aml_active_alerts',
    'Number of active alerts',
    ['severity'],
    registry=registry
)

# System Metrics
SYSTEM_MEMORY_USAGE = Gauge(
    'aml_system_memory_bytes',
    'System memory usage in bytes',
    registry=registry
)

SYSTEM_CPU_USAGE = Gauge(
    'aml_system_cpu_percent',
    'System CPU usage percentage',
    registry=registry
)

PROCESS_MEMORY_USAGE = Gauge(
    'aml_process_memory_bytes',
    'Process memory usage in bytes',
    registry=registry
)

# Graph Metrics
GRAPH_NODES = Gauge(
    'aml_graph_nodes_total',
    'Total number of nodes in graph',
    registry=registry
)

GRAPH_EDGES = Gauge(
    'aml_graph_edges_total',
    'Total number of edges in graph',
    registry=registry
)

GRAPH_LOAD_TIME = Histogram(
    'aml_graph_load_duration_seconds',
    'Graph loading duration in seconds',
    registry=registry
)

# Database Metrics
DB_CONNECTIONS = Gauge(
    'aml_db_connections',
    'Number of active database connections',
    registry=registry
)

DB_QUERY_DURATION = Histogram(
    'aml_db_query_duration_seconds',
    'Database query duration in seconds',
    ['query_type'],
    registry=registry
)

# Redis Metrics
REDIS_CONNECTIONS = Gauge(
    'aml_redis_connections',
    'Number of active Redis connections',
    registry=registry
)

REDIS_MEMORY_USAGE = Gauge(
    'aml_redis_memory_bytes',
    'Redis memory usage in bytes',
    registry=registry
)

def track_request_metrics(method: str, endpoint: str, status: int, duration: float):
    """Track API request metrics"""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

def track_prediction_metrics(model_type: str, framework: str, duration: float, accuracy: float = None):
    """Track model prediction metrics"""
    MODEL_PREDICTIONS.labels(model_type=model_type, framework=framework).inc()
    PREDICTION_DURATION.labels(model_type=model_type).observe(duration)
    if accuracy is not None:
        MODEL_ACCURACY.labels(model_type=model_type).set(accuracy)

def track_cache_metrics(cache_type: str, hit: bool, size: int = None):
    """Track cache metrics"""
    if hit:
        CACHE_HITS.labels(cache_type=cache_type).inc()
    else:
        CACHE_MISSES.labels(cache_type=cache_type).inc()
    
    if size is not None:
        CACHE_SIZE.labels(cache_type=cache_type).set(size)

def track_alert_metrics(severity: str, alert_type: str):
    """Track alert metrics"""
    ALERTS_GENERATED.labels(severity=severity, type=alert_type).inc()
    ACTIVE_ALERTS.labels(severity=severity).inc()

def track_graph_metrics(num_nodes: int, num_edges: int, load_time: float):
    """Track graph metrics"""
    GRAPH_NODES.set(num_nodes)
    GRAPH_EDGES.set(num_edges)
    GRAPH_LOAD_TIME.observe(load_time)

def update_system_metrics():
    """Update system metrics"""
    # Memory usage
    memory = psutil.virtual_memory()
    SYSTEM_MEMORY_USAGE.set(memory.used)
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    SYSTEM_CPU_USAGE.set(cpu_percent)
    
    # Process memory
    process = psutil.Process(os.getpid())
    process_memory = process.memory_info().rss
    PROCESS_MEMORY_USAGE.set(process_memory)

def metrics_middleware(func):
    """Decorator to track request metrics"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        ACTIVE_REQUESTS.inc()
        
        try:
            result = await func(*args, **kwargs)
            status = 200
            return result
        except Exception as e:
            status = 500
            raise
        finally:
            duration = time.time() - start_time
            ACTIVE_REQUESTS.dec()
            
            # Extract method and endpoint from request
            method = "UNKNOWN"
            endpoint = "UNKNOWN"
            
            # Try to get request info from args
            for arg in args:
                if hasattr(arg, 'method'):
                    method = arg.method
                if hasattr(arg, 'url'):
                    endpoint = str(arg.url.path)
            
            track_request_metrics(method, endpoint, status, duration)
    
    return wrapper

def get_metrics():
    """Get all metrics in Prometheus format"""
    update_system_metrics()
    return generate_latest(registry)

class MetricsCollector:
    """Metrics collector for AML Engine"""
    
    def __init__(self):
        self.start_time = time.time()
    
    def record_prediction(self, model_type: str, framework: str, duration: float, accuracy: float = None):
        """Record a model prediction"""
        track_prediction_metrics(model_type, framework, duration, accuracy)
    
    def record_cache_operation(self, cache_type: str, hit: bool, size: int = None):
        """Record a cache operation"""
        track_cache_metrics(cache_type, hit, size)
    
    def record_alert(self, severity: str, alert_type: str):
        """Record an alert generation"""
        track_alert_metrics(severity, alert_type)
    
    def record_graph_load(self, num_nodes: int, num_edges: int, load_time: float):
        """Record graph loading metrics"""
        track_graph_metrics(num_nodes, num_edges, load_time)
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds"""
        return time.time() - self.start_time
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics"""
        return {
            'uptime_seconds': self.get_uptime(),
            'system_memory_bytes': psutil.virtual_memory().used,
            'system_cpu_percent': psutil.cpu_percent(),
            'process_memory_bytes': psutil.Process(os.getpid()).memory_info().rss,
            'active_requests': ACTIVE_REQUESTS._value.get(),
            'active_alerts': sum(ACTIVE_ALERTS._metrics.values()),
            'graph_nodes': GRAPH_NODES._value.get(),
            'graph_edges': GRAPH_EDGES._value.get()
        }

# Global metrics collector instance
metrics_collector = MetricsCollector() 