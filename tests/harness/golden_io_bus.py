#!/usr/bin/env python3
"""
Golden I/O Bus - Logs every API call, resource stats, and emits telemetry
"""

import time
import threading
import json
import psutil
import os
import sys
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from collections import defaultdict, deque
import functools

@dataclass
class APICall:
    """Single API call record"""
    call_id: str
    method_name: str
    args: tuple
    kwargs: dict
    start_time: float
    end_time: Optional[float] = None
    success: bool = True
    result: Any = None
    error: Optional[str] = None
    thread_id: int = 0
    
    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

@dataclass  
class ResourceSnapshot:
    """System resource snapshot at point in time"""
    timestamp: float
    rss_mb: float
    vms_mb: float
    cpu_percent: float
    heap_mb: float  # Process-specific heap if available
    thread_count: int
    file_descriptors: int
    
    @classmethod
    def capture(cls) -> 'ResourceSnapshot':
        """Capture current resource snapshot"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get file descriptor count (Unix only)
        try:
            fd_count = process.num_fds()
        except (AttributeError, psutil.AccessDenied):
            fd_count = 0
            
        return cls(
            timestamp=time.time(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            cpu_percent=process.cpu_percent(),
            heap_mb=memory_info.rss / 1024 / 1024,  # Approximation
            thread_count=process.num_threads(),
            file_descriptors=fd_count
        )

class SpanRecorder:
    """OpenTelemetry-style span recording"""
    
    def __init__(self):
        self.spans: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._span_counter = 0
    
    def start_span(self, name: str, parent_id: Optional[str] = None) -> str:
        """Start a new span and return span ID"""
        with self._lock:
            self._span_counter += 1
            span_id = f"span_{self._span_counter}"
            
            span = {
                "span_id": span_id,
                "parent_id": parent_id,
                "name": name,
                "start_time": time.time(),
                "end_time": None,
                "duration_ms": 0.0,
                "status": "started",
                "attributes": {},
                "events": []
            }
            
            self.spans.append(span)
            return span_id
    
    def end_span(self, span_id: str, status: str = "ok", error: Optional[str] = None):
        """End span with status"""
        with self._lock:
            for span in self.spans:
                if span["span_id"] == span_id:
                    span["end_time"] = time.time()
                    span["duration_ms"] = (span["end_time"] - span["start_time"]) * 1000
                    span["status"] = status
                    if error:
                        span["attributes"]["error"] = error
                    break
    
    def add_span_attribute(self, span_id: str, key: str, value: Any):
        """Add attribute to span"""
        with self._lock:
            for span in self.spans:
                if span["span_id"] == span_id:
                    span["attributes"][key] = value
                    break
    
    def add_span_event(self, span_id: str, name: str, attributes: Optional[Dict] = None):
        """Add event to span"""
        with self._lock:
            for span in self.spans:
                if span["span_id"] == span_id:
                    event = {
                        "name": name,
                        "timestamp": time.time(),
                        "attributes": attributes or {}
                    }
                    span["events"].append(event)
                    break

class MetricCollector:
    """Collect and aggregate metrics"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._lock = threading.Lock()
    
    def record_value(self, metric_name: str, value: float, timestamp: Optional[float] = None):
        """Record a metric value"""
        if timestamp is None:
            timestamp = time.time()
            
        with self._lock:
            self.metrics[metric_name].append((timestamp, value))
    
    def get_percentile(self, metric_name: str, percentile: float) -> float:
        """Get percentile value for metric"""
        with self._lock:
            if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
                return 0.0
                
            values = [v for _, v in self.metrics[metric_name]]
            values.sort()
            
            if len(values) == 1:
                return values[0]
                
            index = int((len(values) - 1) * percentile / 100)
            return values[index]
    
    def get_average(self, metric_name: str) -> float:
        """Get average value for metric"""
        with self._lock:
            if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
                return 0.0
                
            values = [v for _, v in self.metrics[metric_name]]
            return sum(values) / len(values)
    
    def get_count(self, metric_name: str) -> int:
        """Get count of recorded values"""
        with self._lock:
            return len(self.metrics[metric_name])

class GoldenIOBus:
    """Main I/O bus that intercepts and logs all API calls"""
    
    def __init__(self, enable_tracing: bool = True, log_file: Optional[str] = None):
        self.enable_tracing = enable_tracing
        self.log_file = log_file
        
        # Core components
        self.api_calls: List[APICall] = []
        self.resource_snapshots: List[ResourceSnapshot] = []
        self.span_recorder = SpanRecorder()
        self.metric_collector = MetricCollector()
        
        # Threading
        self._lock = threading.Lock()
        self._call_counter = 0
        
        # Resource monitoring thread
        self._monitoring = False
        self._monitor_thread = None
        self._monitor_interval = 1.0  # 1 second
        
        print(f"🚌 Golden I/O Bus initialized (tracing: {enable_tracing})")
    
    def start_monitoring(self, interval: float = 1.0):
        """Start background resource monitoring"""
        self._monitor_interval = interval
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitor_thread.start()
        print(f"📊 Resource monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop background resource monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        print("📊 Resource monitoring stopped")
    
    def _monitor_resources(self):
        """Background resource monitoring loop"""
        while self._monitoring:
            try:
                snapshot = ResourceSnapshot.capture()
                
                with self._lock:
                    self.resource_snapshots.append(snapshot)
                    
                    # Record key metrics
                    self.metric_collector.record_value("rss_mb", snapshot.rss_mb)
                    self.metric_collector.record_value("cpu_percent", snapshot.cpu_percent) 
                    self.metric_collector.record_value("thread_count", snapshot.thread_count)
                    self.metric_collector.record_value("file_descriptors", snapshot.file_descriptors)
                
                time.sleep(self._monitor_interval)
                
            except Exception as e:
                print(f"⚠️ Resource monitoring error: {e}")
                time.sleep(self._monitor_interval)
    
    def wrap_method(self, obj: Any, method_name: str) -> Callable:
        """Wrap a method to log all calls through the bus"""
        original_method = getattr(obj, method_name)
        
        @functools.wraps(original_method)
        def wrapped_method(*args, **kwargs):
            if not self.enable_tracing:
                return original_method(*args, **kwargs)
            
            # Generate call ID
            with self._lock:
                self._call_counter += 1
                call_id = f"call_{self._call_counter}"
            
            # Create API call record
            api_call = APICall(
                call_id=call_id,
                method_name=f"{obj.__class__.__name__}.{method_name}",
                args=args,
                kwargs=kwargs,
                start_time=time.time(),
                thread_id=threading.get_ident()
            )
            
            # Start span
            span_id = self.span_recorder.start_span(api_call.method_name)
            self.span_recorder.add_span_attribute(span_id, "call_id", call_id)
            
            try:
                # Execute original method
                result = original_method(*args, **kwargs)
                
                # Record success
                api_call.end_time = time.time()
                api_call.success = True
                api_call.result = result
                
                # End span successfully
                self.span_recorder.end_span(span_id, "ok")
                self.span_recorder.add_span_attribute(span_id, "result_type", type(result).__name__)
                
                # Record latency metric
                self.metric_collector.record_value(
                    f"{method_name}_latency_ms", 
                    api_call.duration_ms
                )
                
                return result
                
            except Exception as e:
                # Record failure
                api_call.end_time = time.time()
                api_call.success = False
                api_call.error = str(e)
                
                # End span with error
                self.span_recorder.end_span(span_id, "error", str(e))
                
                # Record error metric
                self.metric_collector.record_value(f"{method_name}_errors", 1)
                
                raise
                
            finally:
                # Store API call
                with self._lock:
                    self.api_calls.append(api_call)
                    
                    # Log to file if configured
                    if self.log_file:
                        self._log_api_call(api_call)
        
        return wrapped_method
    
    def _log_api_call(self, api_call: APICall):
        """Log API call to file"""
        try:
            with open(self.log_file, 'a') as f:
                log_entry = {
                    "timestamp": api_call.start_time,
                    "call_id": api_call.call_id,
                    "method": api_call.method_name,
                    "duration_ms": api_call.duration_ms,
                    "success": api_call.success,
                    "error": api_call.error,
                    "thread_id": api_call.thread_id
                }
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"⚠️ Failed to log API call: {e}")
    
    def instrument_object(self, obj: Any, methods: List[str]) -> Any:
        """Instrument multiple methods on an object"""
        for method_name in methods:
            if hasattr(obj, method_name):
                wrapped_method = self.wrap_method(obj, method_name)
                setattr(obj, method_name, wrapped_method)
        return obj
    
    @contextmanager
    def trace_operation(self, operation_name: str):
        """Context manager for tracing high-level operations"""
        span_id = self.span_recorder.start_span(operation_name)
        start_resource = ResourceSnapshot.capture()
        
        try:
            yield span_id
            self.span_recorder.end_span(span_id, "ok")
            
        except Exception as e:
            self.span_recorder.end_span(span_id, "error", str(e))
            raise
            
        finally:
            end_resource = ResourceSnapshot.capture()
            
            # Record resource deltas
            rss_delta = end_resource.rss_mb - start_resource.rss_mb
            self.span_recorder.add_span_attribute(span_id, "rss_delta_mb", rss_delta)
            self.span_recorder.add_span_attribute(span_id, "thread_delta", 
                                                end_resource.thread_count - start_resource.thread_count)
    
    def get_call_statistics(self) -> Dict[str, Any]:
        """Get comprehensive call statistics"""
        with self._lock:
            total_calls = len(self.api_calls)
            successful_calls = sum(1 for call in self.api_calls if call.success)
            failed_calls = total_calls - successful_calls
            
            # Group by method
            method_stats = defaultdict(list)
            for call in self.api_calls:
                method_stats[call.method_name].append(call.duration_ms)
            
            # Calculate percentiles for each method
            method_latencies = {}
            for method, latencies in method_stats.items():
                latencies.sort()
                n = len(latencies)
                if n > 0:
                    method_latencies[method] = {
                        "count": n,
                        "p50": latencies[int(n * 0.5)],
                        "p95": latencies[int(n * 0.95)],
                        "p99": latencies[int(n * 0.99)],
                        "avg": sum(latencies) / n
                    }
            
            return {
                "total_calls": total_calls,
                "successful_calls": successful_calls,
                "failed_calls": failed_calls,
                "success_rate": successful_calls / total_calls if total_calls > 0 else 0.0,
                "method_latencies": method_latencies,
                "resource_snapshots": len(self.resource_snapshots),
                "spans": len(self.span_recorder.spans)
            }
    
    def export_telemetry(self, filename: str):
        """Export all telemetry data to JSON file"""
        telemetry_data = {
            "api_calls": [asdict(call) for call in self.api_calls],
            "resource_snapshots": [asdict(snapshot) for snapshot in self.resource_snapshots],
            "spans": self.span_recorder.spans,
            "statistics": self.get_call_statistics()
        }
        
        with open(filename, 'w') as f:
            json.dump(telemetry_data, f, indent=2)
        
        print(f"📄 Telemetry exported to {filename}")
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_monitoring()
        print("🧹 Golden I/O Bus cleanup complete")

# Global bus instance
_global_bus: Optional[GoldenIOBus] = None

def get_global_bus() -> GoldenIOBus:
    """Get or create global I/O bus"""
    global _global_bus
    if _global_bus is None:
        _global_bus = GoldenIOBus()
    return _global_bus

def setup_instrumentation(obj: Any, methods: List[str]) -> Any:
    """Convenience function to instrument an object"""
    bus = get_global_bus()
    return bus.instrument_object(obj, methods)

@contextmanager
def trace_operation(operation_name: str):
    """Convenience context manager for tracing"""
    bus = get_global_bus()
    with bus.trace_operation(operation_name) as span_id:
        yield span_id