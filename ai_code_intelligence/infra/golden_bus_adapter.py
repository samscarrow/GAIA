#!/usr/bin/env python3
"""
Golden Bus Adapter - Bridge between AI Code Intelligence and GAIA Test Framework
Provides observability, determinism, and measurability to semantic operations
"""

import sys
import os
import time
import functools
from typing import Dict, List, Any, Optional, Callable
from contextlib import contextmanager
import hashlib
import json

# Import GAIA's golden I/O bus
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from tests.harness.golden_io_bus import GoldenIOBus, SpanRecorder, MetricCollector
from tests.harness.test_config import DeterministicRandom, FakeClock, TestConfig, TestProfile

class GoldenBusAdapter:
    """Adapter to instrument AI Code Intelligence with GAIA observability"""
    
    def __init__(self, config: Optional[TestConfig] = None, enable_tracing: bool = True):
        """
        Initialize adapter with optional test configuration
        
        Args:
            config: TestConfig for deterministic execution (None for production)
            enable_tracing: Whether to enable detailed tracing
        """
        self.config = config
        self.enable_tracing = enable_tracing
        
        # Core components from GAIA
        self.bus = GoldenIOBus(enable_tracing=enable_tracing)
        self.span_recorder = self.bus.span_recorder
        self.metric_collector = self.bus.metric_collector
        
        # Deterministic components (if config provided)
        if config:
            self.rng = DeterministicRandom(config.seed)
            self.clock = FakeClock()
            self.config_hash = config.config_hash()
            print(f"🔧 Golden Bus Adapter initialized (deterministic mode)")
            print(f"   Config Hash: {self.config_hash}")
            print(f"   Seed: {config.seed}")
        else:
            self.rng = None
            self.clock = None
            self.config_hash = self._generate_runtime_hash()
            print(f"🔧 Golden Bus Adapter initialized (production mode)")
        
        # Start monitoring
        self.bus.start_monitoring(interval=1.0)
    
    def _generate_runtime_hash(self) -> str:
        """Generate hash for non-deterministic runtime"""
        runtime_info = {
            "timestamp": time.time(),
            "pid": os.getpid(),
            "enable_tracing": self.enable_tracing
        }
        return hashlib.sha256(json.dumps(runtime_info).encode()).hexdigest()[:12]
    
    @contextmanager
    def span(self, name: str, **attributes):
        """
        Create a traced span for semantic operations
        
        Example:
            with adapter.span("activation.run", seed=seed, context=ctx):
                # Activation logic here
                pass
        """
        span_id = self.span_recorder.start_span(name)
        
        # Add attributes
        for key, value in attributes.items():
            self.span_recorder.add_span_attribute(span_id, key, value)
        
        # Add config hash for traceability
        if self.config_hash:
            self.span_recorder.add_span_attribute(span_id, "config_hash", self.config_hash)
        
        start_time = time.time()
        
        try:
            yield span_id
            
            # Record success metrics
            duration_ms = (time.time() - start_time) * 1000
            self.metric_collector.record_value(f"{name}_latency_ms", duration_ms)
            self.span_recorder.end_span(span_id, "ok")
            
        except Exception as e:
            # Record failure metrics
            self.metric_collector.record_value(f"{name}_errors", 1)
            self.span_recorder.end_span(span_id, "error", str(e))
            raise
    
    def record_metric(self, name: str, value: float):
        """Record a metric value"""
        self.metric_collector.record_value(name, value)
    
    def record_semantic_quality(self, recall_at_k: float, ndcg_at_k: float, k: int = 10):
        """Record semantic search quality metrics"""
        self.metric_collector.record_value(f"recall_at_{k}", recall_at_k)
        self.metric_collector.record_value(f"ndcg_at_{k}", ndcg_at_k)
        
        # Check against thresholds
        if recall_at_k < 0.95:
            self.span_recorder.add_span_event(
                "quality_degradation",
                {"recall_at_k": recall_at_k, "threshold": 0.95}
            )
    
    def record_activation(self, start_node: str, activated_count: int, 
                         propagation_depth: int, duration_ms: float):
        """Record activation metrics"""
        self.metric_collector.record_value("activation_nodes", activated_count)
        self.metric_collector.record_value("activation_depth", propagation_depth)
        self.metric_collector.record_value("activation_latency_ms", duration_ms)
        
        # Record as event
        self.bus.span_recorder.add_span_event(
            "activation_complete",
            {
                "start_node": start_node,
                "activated_count": activated_count,
                "depth": propagation_depth,
                "duration_ms": duration_ms
            }
        )
    
    def record_model_switch(self, from_model: str, to_model: str, 
                           accuracy: float, threshold: float = 0.85):
        """Record model fallback events"""
        self.metric_collector.record_value("model_switches", 1)
        self.metric_collector.record_value("model_accuracy", accuracy)
        
        self.bus.span_recorder.add_span_event(
            "model_fallback",
            {
                "from_model": from_model,
                "to_model": to_model,
                "accuracy": accuracy,
                "threshold": threshold
            }
        )
    
    def instrument_semantic_operation(self, func: Callable) -> Callable:
        """
        Decorator to instrument semantic operations
        
        Example:
            @adapter.instrument_semantic_operation
            async def activate_neighborhood(self, node_id, context):
                # Activation logic
                return activated_nodes
        """
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            operation_name = f"{func.__module__}.{func.__name__}"
            
            with self.span(operation_name, args_count=len(args), kwargs_count=len(kwargs)):
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Record success
                    duration_ms = (time.time() - start_time) * 1000
                    self.record_metric(f"{func.__name__}_duration_ms", duration_ms)
                    
                    return result
                    
                except Exception as e:
                    # Record failure
                    self.record_metric(f"{func.__name__}_failures", 1)
                    raise
        
        return wrapper
    
    def get_deterministic_seed(self, base: str) -> int:
        """Get deterministic seed for operations"""
        if self.rng:
            # Use deterministic RNG
            return self.rng.randint(0, 2**31)
        else:
            # Use time-based seed in production
            return int(time.time() * 1000000) % 2**31
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            "config_hash": self.config_hash,
            "deterministic": self.config is not None,
            "bus_stats": self.bus.get_call_statistics(),
            "metrics": {
                "activation_latency_p99": self.metric_collector.get_percentile("activation_latency_ms", 99),
                "recall_at_10_avg": self.metric_collector.get_average("recall_at_10"),
                "ndcg_at_10_avg": self.metric_collector.get_average("ndcg_at_10"),
                "model_switches": self.metric_collector.get_count("model_switches"),
                "total_activations": self.metric_collector.get_count("activation_nodes")
            }
        }
    
    def export_telemetry(self, filename: Optional[str] = None):
        """Export all telemetry data"""
        if filename is None:
            filename = f"ai_telemetry_{self.config_hash}.json"
        
        self.bus.export_telemetry(filename)
        print(f"📊 Telemetry exported to {filename}")
    
    def cleanup(self):
        """Clean up resources"""
        self.bus.cleanup()

# Global adapter instance
_global_adapter: Optional[GoldenBusAdapter] = None

def get_adapter(config: Optional[TestConfig] = None) -> GoldenBusAdapter:
    """Get or create global adapter instance"""
    global _global_adapter
    if _global_adapter is None:
        _global_adapter = GoldenBusAdapter(config)
    return _global_adapter

def span(name: str, **attributes):
    """Convenience decorator for tracing spans"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            adapter = get_adapter()
            with adapter.span(name, **attributes):
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            adapter = get_adapter()
            with adapter.span(name, **attributes):
                return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Example usage functions
def example_cognitive_monitor_integration():
    """Example: How to integrate with live_cognitive_monitor.py"""
    
    # In live_cognitive_monitor.py, replace:
    # async def monitor_activation(self, ...):
    #     # Original logic
    
    # With:
    from infra.golden_bus_adapter import span, get_adapter
    
    @span("cognitive.monitor_activation", component="live_monitor")
    async def monitor_activation(self, thought_id, context):
        adapter = get_adapter()
        
        # Record start
        start_time = time.time()
        
        # Original activation logic here
        activated_nodes = await self.activate_neighborhood(thought_id, context)
        
        # Record metrics
        duration_ms = (time.time() - start_time) * 1000
        adapter.record_activation(
            start_node=thought_id,
            activated_count=len(activated_nodes),
            propagation_depth=3,  # Calculate actual depth
            duration_ms=duration_ms
        )
        
        return activated_nodes

def example_orchestrator_integration():
    """Example: How to integrate with adaptive_orchestrator.py"""
    
    # In adaptive_orchestrator.py:
    from infra.golden_bus_adapter import get_adapter
    
    async def select_model(self, task_type, current_accuracy):
        adapter = get_adapter()
        
        # Model selection logic
        if current_accuracy < 0.85:
            # Record fallback
            adapter.record_model_switch(
                from_model=self.current_model,
                to_model=self.fallback_model,
                accuracy=current_accuracy
            )
            
            return self.fallback_model
        
        return self.current_model

def example_deterministic_test():
    """Example: How to run deterministic tests"""
    
    # Create deterministic config
    config = TestConfig.create(TestProfile.SMALL, seed=42)
    
    # Initialize adapter with config
    adapter = GoldenBusAdapter(config)
    
    # Now all operations are deterministic
    seed = adapter.get_deterministic_seed("activation")
    
    # Use seed for reproducible activations
    # Same seed + same inputs = same outputs
    
    # Export telemetry with config hash
    adapter.export_telemetry()  # Creates ai_telemetry_<hash>.json