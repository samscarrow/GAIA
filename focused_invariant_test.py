#!/usr/bin/env python3
"""
FOCUSED INVARIANT TEST: Memory zones resize within bounds
CRISP PASS/FAIL: Zone size ∈ [100KB, 512KB] - NO EXCEPTIONS
"""

import sys
import os
import asyncio
import time
import numpy as np

sys.path.append('.')
from memory.hierarchical_memory_fixed import HierarchicalMemoryManager
from kernel.enhanced_core import EnhancedGAIAKernel

class FocusedMemoryZoneTest:
    """Focused test with clear pass/fail criteria"""
    
    def __init__(self):
        self.MIN_SIZE = 100 * 1024  # 100KB
        self.MAX_SIZE = 512 * 1024  # 512KB
        self.violations = []
        self.latencies = []
        
    async def test_zone_size_invariant(self) -> dict:
        """
        INVARIANT: Zone size ∈ [100KB, 512KB] 
        APPROACH: Store data, check ALL zones stay in bounds
        PASS/FAIL: Zero violations = PASS, Any violation = FAIL
        """
        print("🧪 FOCUSED TEST: Zone Size Invariant")
        
        # Initialize with small memory to trigger zone management faster
        kernel = EnhancedGAIAKernel(memory_size_mb=32)  # Small to force constraints
        await kernel.initialize()
        
        memory_manager = kernel.memory_manager
        
        try:
            # Test pattern: Store objects of varying sizes
            test_objects = [
                (1000, "small"),      # 1KB objects
                (5000, "medium"),     # 5KB objects  
                (25000, "large"),     # 25KB objects
                (75000, "xlarge"),    # 75KB objects
            ]
            
            allocations = 0
            
            for size, category in test_objects:
                for i in range(20):  # 20 objects per category
                    # Create test data
                    data = np.random.randn(size).astype(np.float32)
                    key = f"test_{category}_{i}"
                    
                    # Measure allocation time
                    start_time = time.time()
                    
                    await memory_manager.store(key, data, semantic_category=category)
                    
                    latency_ms = (time.time() - start_time) * 1000
                    self.latencies.append(latency_ms)
                    allocations += 1
                    
                    # CHECK INVARIANT: All zones must be within bounds
                    self._check_all_zones_in_bounds(memory_manager)
                    
                    if allocations % 10 == 0:
                        print(f"   Allocations: {allocations}, Violations: {len(self.violations)}")
                        
                        # Early exit if violations found
                        if len(self.violations) > 0:
                            print(f"❌ EARLY FAILURE: Violations detected")
                            break
                
                if len(self.violations) > 0:
                    break
            
            # Final invariant check
            self._check_all_zones_in_bounds(memory_manager)
            
            # Calculate metrics
            p99_latency = sorted(self.latencies)[int(len(self.latencies) * 0.99)] if self.latencies else 0.0
            
            result = {
                "total_allocations": allocations,
                "total_violations": len(self.violations),
                "invariant_passed": len(self.violations) == 0,
                "p99_latency_ms": p99_latency,
                "latency_threshold_met": p99_latency <= 2.0,
                "zone_count": len(memory_manager.zones),
                "violations": self.violations[:5],  # First 5 violations
            }
            
            return result
            
        finally:
            # Cleanup if method exists
            if hasattr(kernel, 'cleanup'):
                await kernel.cleanup()
    
    def _check_all_zones_in_bounds(self, memory_manager):
        """Check ALL zones are within size bounds"""
        try:
            for zone_id, zone in memory_manager.zones.items():
                size = zone.size_bytes
                
                if size < self.MIN_SIZE:
                    self.violations.append({
                        "zone_id": zone_id,
                        "violation": "UNDER_MINIMUM",
                        "size_kb": size / 1024,
                        "min_kb": self.MIN_SIZE / 1024,
                        "timestamp": time.time()
                    })
                
                elif size > self.MAX_SIZE:
                    self.violations.append({
                        "zone_id": zone_id,
                        "violation": "OVER_MAXIMUM", 
                        "size_kb": size / 1024,
                        "max_kb": self.MAX_SIZE / 1024,
                        "timestamp": time.time()
                    })
                    
        except Exception as e:
            self.violations.append({
                "zone_id": "unknown",
                "violation": "CHECK_ERROR",
                "error": str(e),
                "timestamp": time.time()
            })

async def run_focused_test():
    """Run the focused invariant test"""
    
    print("🔬 FOCUSED MEMORY ZONE INVARIANT TEST")
    print("=" * 60)
    print("CLAIM: Zone size ∈ [100KB, 512KB]")  
    print("PASS CRITERIA: Zero violations")
    print("FAIL CRITERIA: Any zone outside bounds")
    print("=" * 60)
    
    tester = FocusedMemoryZoneTest()
    
    start_time = time.time()
    result = await tester.test_zone_size_invariant()
    duration = time.time() - start_time
    
    # CRISP PASS/FAIL DECISION
    invariant_pass = result["invariant_passed"]
    latency_pass = result["latency_threshold_met"]
    
    overall_pass = invariant_pass and latency_pass
    
    # REPORT RESULTS
    print(f"\n📊 TEST RESULTS ({duration:.2f}s)")
    print(f"   Allocations: {result['total_allocations']}")
    print(f"   Zone Count: {result['zone_count']}")
    print(f"   P99 Latency: {result['p99_latency_ms']:.2f}ms")
    
    print(f"\n🎯 PASS/FAIL GATES:")
    print(f"   ✅ Zone Size Invariant: {'PASS' if invariant_pass else 'FAIL'} ({result['total_violations']} violations)")
    print(f"   ✅ Latency SLO (≤2ms): {'PASS' if latency_pass else 'FAIL'} ({result['p99_latency_ms']:.2f}ms)")
    
    if result["violations"]:
        print(f"\n❌ VIOLATIONS FOUND:")
        for v in result["violations"]:
            print(f"   • {v['zone_id']}: {v['violation']} - {v.get('size_kb', 'N/A'):.1f}KB")
    
    print(f"\n🏆 FINAL VERDICT: {'✅ PASS' if overall_pass else '❌ FAIL'}")
    
    if not overall_pass:
        print(f"\n🔧 SYSTEM NOT READY:")
        if not invariant_pass:
            print(f"   • Zone size invariant violated - architectural fix required")
        if not latency_pass:  
            print(f"   • Latency SLO missed - performance optimization needed")
    else:
        print(f"\n🟢 SYSTEM READY: All invariants verified under test conditions")
    
    return overall_pass

if __name__ == "__main__":
    passed = asyncio.run(run_focused_test())
    exit(0 if passed else 1)