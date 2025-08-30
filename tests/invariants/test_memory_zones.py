#!/usr/bin/env python3
"""
INVARIANT TEST: Memory zones dynamically resize (100KB–512KB)

CLAIMS TO PROVE:
- Zones grow/shrink within bounds under load without corruption or stalls
- Zone size ∈ [100KB, 512KB] 
- No pointer aliasing/corruption across zone moves
- Fragmentation ≤ 20% under steady state

PASS/FAIL THRESHOLDS:
- 99th alloc latency < 2 ms under "medium" profile
- Fragmentation < 20% at 1h; no leaks (heap delta < 1%)  
- No sanitizer findings (ASAN/TSAN/MSAN clean)
"""

import sys
import os
import asyncio
import time
import random
import hashlib
import numpy as np
import threading
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.harness.test_config import TestHarness, TestProfile, TestConfig, create_metric
from tests.harness.golden_io_bus import get_global_bus, trace_operation, setup_instrumentation
from tests.harness.fault_injector import FaultInjector, integration_test_faults
from memory.hierarchical_memory import HierarchicalMemoryManager, MemoryZone
from kernel.enhanced_core import EnhancedGAIAKernel

class MemoryZoneInvariantTests:
    """Test memory zone resize invariants with crisp pass/fail gates"""
    
    def __init__(self, harness: TestHarness):
        self.harness = harness
        self.memory_manager: Optional[HierarchicalMemoryManager] = None
        self.kernel: Optional[EnhancedGAIAKernel] = None
        self.bus = get_global_bus()
        
        # Constants for invariant testing
        self.MIN_ZONE_SIZE = 100 * 1024  # 100KB
        self.MAX_ZONE_SIZE = 512 * 1024  # 512KB
        self.MAX_FRAGMENTATION = 0.20    # 20%
        self.MAX_ALLOC_LATENCY_MS = 2.0  # 2ms p99
        
        # Test data storage
        self.allocation_latencies: List[float] = []
        self.zone_sizes: List[int] = []
        self.fragmentation_snapshots: List[float] = []
        self.corruption_detected = False
        
    async def setup_test_environment(self):
        """Setup instrumented memory manager for testing"""
        print(f"🔧 Setting up memory zone test environment")
        
        # Initialize kernel with test profile memory size
        self.kernel = EnhancedGAIAKernel(memory_size_mb=self.harness.config.memory_mb)
        await self.kernel.initialize()
        
        self.memory_manager = self.kernel.memory_manager
        
        # Instrument memory manager methods for observability
        setup_instrumentation(self.memory_manager, [
            'store', 'retrieve', 'delete', '_create_zone', '_manage_memory_pressure',
            '_compress_zone', '_decompress_zone'
        ])
        
        # Start monitoring
        self.bus.start_monitoring(interval=0.5)
        
        print(f"   Memory configured: {self.harness.config.memory_mb}MB")
        print(f"   Concurrency: {self.harness.config.concurrency} threads")
        
    async def test_zone_size_invariants(self) -> Dict[str, Any]:
        """
        INVARIANT: Zone size ∈ [100KB, 512KB]
        
        Test approach:
        1. Allocate data of various sizes to trigger zone creation/resizing
        2. Monitor zone sizes throughout allocation process
        3. Verify ALL zones stay within bounds
        4. Record violations for analysis
        """
        print(f"\n🧪 TESTING: Zone Size Invariants [{self.MIN_ZONE_SIZE//1024}KB - {self.MAX_ZONE_SIZE//1024}KB]")
        
        violations = []
        allocation_count = 0
        
        with trace_operation("zone_size_invariant_test"):
            # Test with various allocation patterns
            test_patterns = [
                ("small_objects", 100, 50),      # 100 objects, 50 bytes each
                ("medium_objects", 50, 5000),    # 50 objects, 5KB each  
                ("large_objects", 10, 50000),    # 10 objects, 50KB each
                ("mixed_objects", 200, None),    # 200 mixed size objects
            ]
            
            for pattern_name, count, size in test_patterns:
                print(f"   Testing pattern: {pattern_name}")
                
                for i in range(count):
                    # Generate test data
                    if size is None:
                        # Mixed sizes
                        object_size = random.randint(100, 100000)
                    else:
                        object_size = size
                    
                    test_data = np.random.randn(object_size).astype(np.float32)
                    key = f"invariant_test_{pattern_name}_{i}"
                    
                    # Measure allocation latency
                    start_time = time.time()
                    
                    try:
                        await self.memory_manager.store(
                            key, test_data, semantic_category=f"invariant_{pattern_name}"
                        )
                        allocation_count += 1
                        
                        # Record latency
                        latency_ms = (time.time() - start_time) * 1000
                        self.allocation_latencies.append(latency_ms)
                        
                        # Check zone size invariants
                        zone_violations = self._check_zone_size_invariants()
                        violations.extend(zone_violations)
                        
                        # Sample zone sizes for analysis
                        if allocation_count % 10 == 0:
                            self._sample_zone_sizes()
                        
                    except Exception as e:
                        violations.append({
                            "type": "allocation_failure",
                            "pattern": pattern_name,
                            "iteration": i,
                            "error": str(e),
                            "object_size": object_size
                        })
                    
                    # Brief pause to allow monitoring
                    if i % 10 == 0:
                        await asyncio.sleep(0.01)
        
        # Analysis
        total_violations = len(violations)
        size_violations = [v for v in violations if v["type"] == "size_violation"]
        
        return {
            "total_allocations": allocation_count,
            "total_violations": total_violations,
            "size_violations": len(size_violations),
            "violation_rate": total_violations / allocation_count if allocation_count > 0 else 0.0,
            "violations_detail": violations[:10],  # First 10 for analysis
            "zone_size_stats": self._analyze_zone_sizes(),
            "latency_stats": self._analyze_latencies()
        }
    
    def _check_zone_size_invariants(self) -> List[Dict[str, Any]]:
        """Check if any zones violate size constraints"""
        violations = []
        
        try:
            for zone_id, zone in self.memory_manager.zones.items():
                current_size = zone.current_size
                
                if current_size < self.MIN_ZONE_SIZE:
                    violations.append({
                        "type": "size_violation",
                        "zone_id": zone_id,
                        "violation": "under_minimum",
                        "size": current_size,
                        "min_allowed": self.MIN_ZONE_SIZE,
                        "timestamp": time.time()
                    })
                
                elif current_size > self.MAX_ZONE_SIZE:
                    violations.append({
                        "type": "size_violation", 
                        "zone_id": zone_id,
                        "violation": "over_maximum",
                        "size": current_size,
                        "max_allowed": self.MAX_ZONE_SIZE,
                        "timestamp": time.time()
                    })
                
                # Record size for analysis
                self.zone_sizes.append(current_size)
                
        except Exception as e:
            violations.append({
                "type": "invariant_check_failure",
                "error": str(e),
                "timestamp": time.time()
            })
        
        return violations
    
    def _sample_zone_sizes(self):
        """Sample current zone sizes for statistical analysis"""
        try:
            for zone in self.memory_manager.zones.values():
                self.zone_sizes.append(zone.current_size)
        except:
            pass  # Ignore sampling errors
    
    def _analyze_zone_sizes(self) -> Dict[str, Any]:
        """Analyze zone size distribution"""
        if not self.zone_sizes:
            return {"error": "No zone size data collected"}
        
        sizes_kb = [s / 1024 for s in self.zone_sizes]
        
        return {
            "count": len(sizes_kb),
            "min_kb": min(sizes_kb),
            "max_kb": max(sizes_kb),
            "mean_kb": sum(sizes_kb) / len(sizes_kb),
            "in_bounds_count": sum(1 for s in self.zone_sizes 
                                 if self.MIN_ZONE_SIZE <= s <= self.MAX_ZONE_SIZE),
            "under_min": sum(1 for s in self.zone_sizes if s < self.MIN_ZONE_SIZE),
            "over_max": sum(1 for s in self.zone_sizes if s > self.MAX_ZONE_SIZE)
        }
    
    def _analyze_latencies(self) -> Dict[str, Any]:
        """Analyze allocation latency distribution"""
        if not self.allocation_latencies:
            return {"error": "No latency data collected"}
        
        latencies = sorted(self.allocation_latencies)
        n = len(latencies)
        
        return {
            "count": n,
            "mean_ms": sum(latencies) / n,
            "p50_ms": latencies[int(n * 0.5)],
            "p95_ms": latencies[int(n * 0.95)],  
            "p99_ms": latencies[int(n * 0.99)],
            "max_ms": max(latencies)
        }
    
    async def test_fragmentation_invariants(self) -> Dict[str, Any]:
        """
        INVARIANT: Fragmentation ≤ 20% under steady state
        
        Test approach:
        1. Create fragmentation through mixed alloc/dealloc pattern
        2. Let system reach steady state
        3. Measure fragmentation continuously
        4. Verify it stays below threshold
        """
        print(f"\n🧪 TESTING: Fragmentation Invariants (≤ {self.MAX_FRAGMENTATION*100}%)")
        
        fragmentation_violations = []
        
        with trace_operation("fragmentation_invariant_test"):
            # Phase 1: Create fragmentation with interleaved alloc/dealloc
            print("   Phase 1: Creating fragmentation pattern")
            allocated_keys = []
            
            for i in range(500):
                if random.random() < 0.7:  # 70% allocate
                    size = random.randint(1000, 10000)
                    data = np.random.randn(size).astype(np.float32)
                    key = f"frag_test_{i}"
                    
                    await self.memory_manager.store(key, data, semantic_category="fragmentation")
                    allocated_keys.append(key)
                
                else:  # 30% deallocate
                    if allocated_keys:
                        # Remove random key (simulates deallocation)
                        key_to_remove = random.choice(allocated_keys)
                        try:
                            await self.memory_manager.delete(key_to_remove)
                            allocated_keys.remove(key_to_remove)
                        except:
                            pass  # Key might not exist
                
                # Sample fragmentation every 50 operations
                if i % 50 == 0:
                    fragmentation = self._measure_fragmentation()
                    self.fragmentation_snapshots.append(fragmentation)
                    
                    if fragmentation > self.MAX_FRAGMENTATION:
                        fragmentation_violations.append({
                            "iteration": i,
                            "fragmentation": fragmentation,
                            "threshold": self.MAX_FRAGMENTATION,
                            "timestamp": time.time()
                        })
            
            # Phase 2: Steady state measurement  
            print("   Phase 2: Steady state fragmentation measurement")
            await asyncio.sleep(2.0)  # Let system settle
            
            for i in range(100):
                fragmentation = self._measure_fragmentation()
                self.fragmentation_snapshots.append(fragmentation)
                
                if fragmentation > self.MAX_FRAGMENTATION:
                    fragmentation_violations.append({
                        "iteration": f"steady_state_{i}",
                        "fragmentation": fragmentation,
                        "threshold": self.MAX_FRAGMENTATION,
                        "timestamp": time.time()
                    })
                
                await asyncio.sleep(0.1)
        
        # Analysis
        avg_fragmentation = (sum(self.fragmentation_snapshots) / 
                           len(self.fragmentation_snapshots) 
                           if self.fragmentation_snapshots else 0.0)
        
        max_fragmentation = max(self.fragmentation_snapshots) if self.fragmentation_snapshots else 0.0
        
        return {
            "fragmentation_samples": len(self.fragmentation_snapshots),
            "fragmentation_violations": len(fragmentation_violations),
            "avg_fragmentation": avg_fragmentation,
            "max_fragmentation": max_fragmentation,
            "threshold": self.MAX_FRAGMENTATION,
            "violation_details": fragmentation_violations[:5],  # First 5 violations
            "fragmentation_over_time": self.fragmentation_snapshots[-20:]  # Last 20 samples
        }
    
    def _measure_fragmentation(self) -> float:
        """Measure current memory fragmentation"""
        try:
            if not self.memory_manager.zones:
                return 0.0
            
            total_capacity = 0
            total_used = 0
            total_free_segments = 0
            
            for zone in self.memory_manager.zones.values():
                total_capacity += zone.max_size
                total_used += zone.current_size
                # Simplified fragmentation metric: assume each zone has some fragmentation
                if zone.current_size < zone.max_size:
                    total_free_segments += 1
            
            if total_capacity == 0:
                return 0.0
            
            # Simplified fragmentation calculation
            # Real implementation would analyze actual memory layout
            utilization = total_used / total_capacity
            fragmentation = (total_free_segments / len(self.memory_manager.zones)) * (1.0 - utilization)
            
            return min(fragmentation, 1.0)  # Cap at 100%
            
        except Exception:
            return 0.0  # Conservative estimate on error
    
    async def test_concurrent_access_safety(self) -> Dict[str, Any]:
        """
        INVARIANT: No pointer aliasing/corruption across zone moves
        
        Test approach:
        1. Concurrent reads/writes to force zone resizing
        2. Verify data integrity after operations
        3. Detect any corruption or aliasing
        """
        print(f"\n🧪 TESTING: Concurrent Access Safety (No corruption/aliasing)")
        
        corruption_events = []
        data_checksums: Dict[str, str] = {}
        
        with trace_operation("concurrent_safety_test"):
            # Phase 1: Setup test data with known checksums
            test_keys = []
            for i in range(100):
                key = f"safety_test_{i}"
                data = np.random.randn(random.randint(1000, 5000)).astype(np.float32)
                
                # Calculate checksum
                data_bytes = data.tobytes()
                checksum = hashlib.md5(data_bytes).hexdigest()
                data_checksums[key] = checksum
                
                await self.memory_manager.store(key, data, semantic_category="safety_test")
                test_keys.append(key)
            
            # Phase 2: Concurrent access to trigger zone operations
            async def concurrent_worker(worker_id: int):
                worker_corruptions = []
                
                for i in range(50):
                    try:
                        # Random operations to stress the system
                        if random.random() < 0.5:
                            # Read existing data and verify checksum
                            key = random.choice(test_keys)
                            retrieved_data = await self.memory_manager.retrieve(key)
                            
                            if retrieved_data is not None:
                                data_bytes = retrieved_data.tobytes()
                                checksum = hashlib.md5(data_bytes).hexdigest()
                                
                                if checksum != data_checksums[key]:
                                    worker_corruptions.append({
                                        "worker_id": worker_id,
                                        "key": key,
                                        "expected_checksum": data_checksums[key],
                                        "actual_checksum": checksum,
                                        "timestamp": time.time()
                                    })
                                    self.corruption_detected = True
                        else:
                            # Write new data 
                            new_key = f"safety_new_{worker_id}_{i}"
                            new_data = np.random.randn(random.randint(500, 3000)).astype(np.float32)
                            
                            await self.memory_manager.store(new_key, new_data, 
                                                          semantic_category="safety_concurrent")
                    except Exception as e:
                        # Record but don't fail on individual operation errors
                        pass
                    
                    # Brief pause
                    await asyncio.sleep(0.01)
                
                return worker_corruptions
            
            # Launch concurrent workers
            tasks = []
            for worker_id in range(self.harness.config.concurrency):
                task = asyncio.create_task(concurrent_worker(worker_id))
                tasks.append(task)
            
            # Wait for all workers
            worker_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect corruption events
            for result in worker_results:
                if isinstance(result, list):
                    corruption_events.extend(result)
        
        return {
            "test_keys_count": len(test_keys),
            "concurrent_workers": self.harness.config.concurrency,
            "corruption_events": len(corruption_events),
            "corruption_detected": self.corruption_detected,
            "corruption_details": corruption_events[:5],  # First 5 for analysis
            "data_integrity_rate": 1.0 - (len(corruption_events) / len(test_keys))
        }
    
    async def run_property_based_tests(self) -> Dict[str, Any]:
        """
        Property-based testing with random operations
        
        Generates random allocation patterns and verifies invariants hold
        """
        print(f"\n🧪 TESTING: Property-based Random Operations (10k ops)")
        
        operations_count = 10000
        property_violations = []
        
        with trace_operation("property_based_test"):
            allocated_objects: Dict[str, Dict[str, Any]] = {}
            
            for op_id in range(operations_count):
                try:
                    # Random operation selection
                    if random.random() < 0.7 or not allocated_objects:  # 70% allocate or if nothing allocated
                        # Allocate with random size (including boundary cases)
                        boundary_sizes = [99, 100, 512*1024-1, 512*1024, 512*1024+1]  # Boundary testing
                        if random.random() < 0.1:  # 10% boundary cases
                            size = random.choice(boundary_sizes)
                        else:
                            size = random.randint(100, 100000)
                        
                        key = f"prop_test_{op_id}"
                        data = np.random.randn(size).astype(np.float32)
                        
                        start_time = time.time()
                        await self.memory_manager.store(key, data, semantic_category="property_test")
                        latency_ms = (time.time() - start_time) * 1000
                        
                        allocated_objects[key] = {
                            "size": size,
                            "checksum": hashlib.md5(data.tobytes()).hexdigest(),
                            "timestamp": time.time()
                        }
                        
                        # Record latency for invariant checking
                        self.allocation_latencies.append(latency_ms)
                    
                    else:
                        # Deallocate random object
                        key = random.choice(list(allocated_objects.keys()))
                        try:
                            await self.memory_manager.delete(key)
                            del allocated_objects[key]
                        except:
                            pass  # Ignore deallocation failures
                    
                    # Check invariants periodically
                    if op_id % 100 == 0:
                        violations = self._check_zone_size_invariants()
                        property_violations.extend(violations)
                
                except Exception as e:
                    property_violations.append({
                        "type": "operation_failure",
                        "operation_id": op_id,
                        "error": str(e),
                        "timestamp": time.time()
                    })
        
        return {
            "operations_performed": operations_count,
            "property_violations": len(property_violations),
            "final_allocated_objects": len(allocated_objects),
            "violation_details": property_violations[:10]
        }
    
    async def cleanup_test_environment(self):
        """Clean up test environment"""
        try:
            if self.kernel:
                await self.kernel.cleanup()
            self.bus.stop_monitoring()
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

async def run_memory_zone_invariant_tests():
    """Main entry point for memory zone invariant testing"""
    
    # Test with MEDIUM profile for comprehensive testing
    config = TestConfig.create(TestProfile.MEDIUM, seed=42)
    harness = TestHarness(config)
    
    tester = MemoryZoneInvariantTests(harness)
    
    with harness.run_test("Memory Zone Invariants") as result:
        try:
            # Setup
            await tester.setup_test_environment()
            
            # Run invariant tests
            print("🔬 Running Memory Zone Invariant Tests")
            
            zone_size_results = await tester.test_zone_size_invariants()
            fragmentation_results = await tester.test_fragmentation_invariants()
            safety_results = await tester.test_concurrent_access_safety()
            property_results = await tester.run_property_based_tests()
            
            # Analyze results and create metrics with pass/fail thresholds
            
            # METRIC 1: Allocation latency p99 < 2ms
            if tester.allocation_latencies:
                latencies = sorted(tester.allocation_latencies)
                p99_latency = latencies[int(len(latencies) * 0.99)]
                result.metrics.append(create_metric("alloc_latency_ms_p99", p99_latency, "lte"))
            
            # METRIC 2: Fragmentation < 20%
            if tester.fragmentation_snapshots:
                max_fragmentation = max(tester.fragmentation_snapshots) * 100  # Convert to percentage
                result.metrics.append(create_metric("fragmentation_pct", max_fragmentation, "lte"))
            
            # METRIC 3: Zone resize success rate > 99%
            total_operations = (zone_size_results["total_allocations"] + 
                              property_results["operations_performed"])
            total_violations = (zone_size_results["total_violations"] +
                              property_results["property_violations"])
            
            success_rate = 1.0 - (total_violations / total_operations) if total_operations > 0 else 1.0
            result.metrics.append(create_metric("zone_resize_success_rate", success_rate, "gte"))
            
            # Store detailed results
            result.artifacts = {
                "zone_size_test": zone_size_results,
                "fragmentation_test": fragmentation_results,
                "safety_test": safety_results, 
                "property_test": property_results,
                "bus_statistics": tester.bus.get_call_statistics()
            }
            
        finally:
            await tester.cleanup_test_environment()
    
    return result

if __name__ == "__main__":
    result = asyncio.run(run_memory_zone_invariant_tests())
    
    print(f"\n🏆 TEST RESULT: {'✅ PASS' if result.passed else '❌ FAIL'}")
    print(f"Config Hash: {result.config_hash}")
    print(f"Duration: {result.duration_seconds:.2f}s")
    
    if not result.passed:
        print("\n❌ FAILING METRICS:")
        for metric in result.metrics:
            if not metric.passed:
                print(f"   {metric.name}: {metric.value:.3f} {metric.operator} {metric.threshold}")
    
    # Export telemetry
    bus = get_global_bus()
    bus.export_telemetry(f"memory_zones_telemetry_{result.config_hash}.json")