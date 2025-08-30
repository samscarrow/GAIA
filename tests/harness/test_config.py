#!/usr/bin/env python3
"""
Test Harness - Determinism, observability, and reproducible execution
"""

import os
import sys
import time
import random
import hashlib
import json
import threading
from typing import Dict, List, Any, Optional, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from contextlib import contextmanager

class TestProfile(Enum):
    """Test execution profiles"""
    SMALL = "small"      # 128MB, 1-4 threads, 5min
    MEDIUM = "medium"    # 2GB, 4-16 threads, 1hr  
    LARGE = "large"      # 16GB, 16-64 threads, 24hr

@dataclass
class TestConfig:
    """Deterministic test configuration"""
    profile: TestProfile
    seed: int
    duration_seconds: int
    concurrency: int
    memory_mb: int
    enable_faults: bool = False
    enable_tracing: bool = True
    
    @classmethod
    def create(cls, profile: TestProfile, seed: Optional[int] = None) -> 'TestConfig':
        """Create config with profile defaults"""
        if seed is None:
            seed = int(time.time() * 1000000) % 2**31
            
        config_map = {
            TestProfile.SMALL: cls(
                profile=profile,
                seed=seed,
                duration_seconds=300,  # 5 minutes
                concurrency=4,
                memory_mb=128,
            ),
            TestProfile.MEDIUM: cls(
                profile=profile,
                seed=seed,
                duration_seconds=3600,  # 1 hour
                concurrency=16,
                memory_mb=2048,
            ),
            TestProfile.LARGE: cls(
                profile=profile,
                seed=seed,
                duration_seconds=86400,  # 24 hours
                concurrency=64,
                memory_mb=16384,
            )
        }
        return config_map[profile]
    
    def config_hash(self) -> str:
        """Generate reproducible hash of config"""
        config_dict = asdict(self)
        # Convert enum to string for JSON serialization
        config_dict['profile'] = config_dict['profile'].value
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]

class DeterministicRandom:
    """Deterministic random number generator"""
    
    def __init__(self, seed: int):
        self.seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)
        
    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)
    
    def random(self) -> float:
        return self._rng.random()
        
    def choice(self, seq):
        return self._rng.choice(seq)
        
    def randn(self, *shape) -> np.ndarray:
        return self._np_rng.randn(*shape)
        
    def randrange(self, *args) -> int:
        return self._rng.randrange(*args)

class FakeClock:
    """Deterministic time control for testing"""
    
    def __init__(self, start_time: float = 1640995200.0):  # 2022-01-01
        self._current_time = start_time
        self._lock = threading.Lock()
        
    def time(self) -> float:
        with self._lock:
            return self._current_time
            
    def advance(self, seconds: float):
        with self._lock:
            self._current_time += seconds
            
    def sleep(self, seconds: float):
        self.advance(seconds)
        
    @contextmanager
    def patch_time(self):
        """Context manager to patch time.time() globally"""
        import time as time_module
        original_time = time_module.time
        original_sleep = time_module.sleep
        
        time_module.time = self.time
        time_module.sleep = self.sleep
        
        try:
            yield self
        finally:
            time_module.time = original_time
            time_module.sleep = original_sleep

@dataclass 
class TestMetric:
    """Single test metric with threshold"""
    name: str
    value: float
    threshold: float
    operator: str  # "lte", "gte", "eq"
    passed: bool
    
    def check(self) -> bool:
        """Check if metric passes threshold"""
        if self.operator == "lte":
            self.passed = self.value <= self.threshold
        elif self.operator == "gte":
            self.passed = self.value >= self.threshold
        elif self.operator == "eq":
            self.passed = abs(self.value - self.threshold) < 1e-6
        else:
            raise ValueError(f"Unknown operator: {self.operator}")
        return self.passed

@dataclass
class TestResult:
    """Complete test execution result"""
    test_name: str
    config: TestConfig
    config_hash: str
    start_time: float
    duration_seconds: float
    success: bool
    metrics: List[TestMetric]
    errors: List[str]
    artifacts: Dict[str, Any]  # Logs, traces, etc.
    
    @property
    def passed(self) -> bool:
        """Overall pass/fail based on metrics and errors"""
        return self.success and all(m.passed for m in self.metrics) and len(self.errors) == 0

class TestHarness:
    """Main test harness orchestrator"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.config_hash = config.config_hash()
        self.rng = DeterministicRandom(config.seed)
        self.clock = FakeClock()
        
        # Setup deterministic environment
        random.seed(config.seed)
        np.random.seed(config.seed)
        os.environ['PYTHONHASHSEED'] = str(config.seed)
        
        print(f"🔧 Test Harness initialized")
        print(f"   Profile: {config.profile.value}")  
        print(f"   Seed: {config.seed}")
        print(f"   Config Hash: {self.config_hash}")
        print(f"   Duration: {config.duration_seconds}s")
        print(f"   Concurrency: {config.concurrency}")
        print(f"   Memory: {config.memory_mb}MB")
    
    def create_test_result(self, test_name: str) -> TestResult:
        """Create empty test result template"""
        return TestResult(
            test_name=test_name,
            config=self.config,
            config_hash=self.config_hash,
            start_time=time.time(),
            duration_seconds=0.0,
            success=False,
            metrics=[],
            errors=[],
            artifacts={}
        )
    
    @contextmanager
    def run_test(self, test_name: str):
        """Context manager for running individual tests"""
        print(f"\n🧪 Running test: {test_name}")
        print("=" * 60)
        
        result = self.create_test_result(test_name)
        start_time = time.time()
        
        try:
            with self.clock.patch_time():
                yield result
                result.success = True
                
        except Exception as e:
            result.errors.append(str(e))
            result.success = False
            print(f"❌ Test failed: {e}")
            
        finally:
            result.duration_seconds = time.time() - start_time
            
            # Check all metrics
            for metric in result.metrics:
                metric.check()
            
            # Print results
            self._print_test_result(result)
    
    def _print_test_result(self, result: TestResult):
        """Print formatted test results"""
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"\n{status} {result.test_name} ({result.duration_seconds:.2f}s)")
        
        if result.metrics:
            print("📊 Metrics:")
            for metric in result.metrics:
                status_icon = "✅" if metric.passed else "❌"
                print(f"   {status_icon} {metric.name}: {metric.value:.3f} {metric.operator} {metric.threshold}")
        
        if result.errors:
            print("❌ Errors:")
            for error in result.errors:
                print(f"   • {error}")

# Test thresholds - these define our SLOs
THRESHOLDS = {
    # Memory zone management
    "alloc_latency_ms_p99": {"lte": 2.0},
    "fragmentation_pct": {"lte": 20.0},
    "zone_resize_success_rate": {"gte": 0.99},
    
    # B-tree semantic search  
    "recall_at_10": {"gte": 0.95},
    "ndcg_at_10": {"gte": 0.97},
    "query_latency_ms_p99": {"lte": 10.0},  # medium profile
    "btree_height": {"lte": 5.0},  # for 10M entries
    
    # Compression
    "compression_trigger_seconds": {"lte": 1.0},
    "compression_ratio": {"gte": 1.5},
    "tail_degradation_pct_p99": {"lte": 30.0},
    "compression_thrash_per_10min": {"lte": 2.0},
    
    # Model fallback
    "accuracy_detection_windows": {"lte": 2.0},
    "failover_mttr_seconds": {"lte": 60.0},
    "flap_stability_pct": {"gte": 90.0},
    
    # Priority interrupts
    "preemption_latency_ms": {"lte": 20.0},
    "missed_deadlines": {"eq": 0.0},
    "starvation_recovery_ms": {"lte": 500.0},
    
    # Checkpointing
    "checkpoint_rpo_seconds": {"lte": 10.0},  # Based on cadence
    "resume_time_ms_p99": {"lte": 20000.0},  # 20s
    "corruption_silent_failures": {"eq": 0.0},
    
    # Background maintenance  
    "maintenance_cpu_pct": {"lte": 30.0},
    "maintenance_suspend_ms": {"lte": 50.0},
    "maintenance_tail_spike_pct": {"lte": 25.0}
}

def get_threshold(metric_name: str, operator: str) -> float:
    """Get threshold value for metric"""
    if metric_name not in THRESHOLDS:
        raise ValueError(f"Unknown metric: {metric_name}")
    
    threshold_config = THRESHOLDS[metric_name]
    if operator not in threshold_config:
        raise ValueError(f"Metric {metric_name} doesn't support operator {operator}")
    
    return threshold_config[operator]

def create_metric(name: str, value: float, operator: str) -> TestMetric:
    """Create metric with automatic threshold lookup"""
    threshold = get_threshold(name, operator)
    return TestMetric(
        name=name,
        value=value,
        threshold=threshold,
        operator=operator,
        passed=False  # Will be set by check()
    )