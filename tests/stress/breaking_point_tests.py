#!/usr/bin/env python3
"""
Breaking Point Tests - Actually find where GAIA breaks
Unlike the gentle "chaos" tests, these are designed to FAIL and find limits
"""

import asyncio
import time
import psutil
import threading
import ctypes
import os
import signal
import resource
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kernel.enhanced_core import EnhancedGAIAKernel

class BreakingPointType(Enum):
    """Different ways to break the system"""
    MEMORY_EXHAUSTION = "memory_exhaustion"
    CPU_SATURATION = "cpu_saturation" 
    DEADLOCK_CREATION = "deadlock_creation"
    STACK_OVERFLOW = "stack_overflow"
    FILE_DESCRIPTOR_EXHAUSTION = "fd_exhaustion"
    THREAD_LIMIT_BREACH = "thread_limit"
    INTERRUPT_STORM = "interrupt_storm"

@dataclass
class BreakingPoint:
    """Records where the system actually broke"""
    test_type: BreakingPointType
    breaking_value: float  # The value where it broke
    symptoms: List[str]
    system_state: Dict[str, Any]
    recovery_possible: bool
    error_message: str

class MemoryExhaustionTest:
    """Actually exhaust memory until system breaks"""
    
    def __init__(self, kernel: EnhancedGAIAKernel):
        self.kernel = kernel
        self.memory_bombs: List[np.ndarray] = []
        
    async def find_memory_breaking_point(self) -> BreakingPoint:
        """Keep allocating until OOM or system crash"""
        print("💣 MEMORY EXHAUSTION - Allocating until system breaks")
        
        allocation_size_mb = 10
        total_allocated = 0
        symptoms = []
        
        try:
            while True:
                # Check current memory usage
                process = psutil.Process()
                mem_info = process.memory_info()
                system_mem = psutil.virtual_memory()
                
                print(f"  Allocated: {total_allocated}MB | RSS: {mem_info.rss//1024//1024}MB | Available: {system_mem.available//1024//1024}MB")
                
                # Detect symptoms
                if system_mem.percent > 90:
                    symptoms.append("System memory >90%")
                if mem_info.rss > 1024 * 1024 * 1024:  # 1GB
                    symptoms.append("Process RSS >1GB")
                    
                # Try to allocate large chunk
                try:
                    # Force actual memory allocation, not just virtual
                    bomb = np.ones((allocation_size_mb * 1024 * 256,), dtype=np.float32)
                    bomb.fill(42)  # Force physical allocation
                    self.memory_bombs.append(bomb)
                    total_allocated += allocation_size_mb
                    
                    # Try to use kernel with this memory pressure
                    await self.kernel.memory_manager.store(
                        f"pressure_test_{total_allocated}",
                        np.random.randn(1000).astype(np.float32),
                        semantic_category="pressure_test"
                    )
                    
                except MemoryError as e:
                    return BreakingPoint(
                        test_type=BreakingPointType.MEMORY_EXHAUSTION,
                        breaking_value=total_allocated,
                        symptoms=symptoms,
                        system_state={
                            "rss_mb": mem_info.rss // 1024 // 1024,
                            "system_memory_percent": system_mem.percent,
                            "available_mb": system_mem.available // 1024 // 1024
                        },
                        recovery_possible=True,
                        error_message=str(e)
                    )
                
                # Increase allocation rate as we get closer to limits
                if system_mem.percent > 80:
                    allocation_size_mb = 50  # Accelerate
                if system_mem.percent > 95:
                    allocation_size_mb = 100  # Ram it home
                    
                # Safety valve - don't kill the host
                if total_allocated > 8000:  # 8GB limit
                    return BreakingPoint(
                        test_type=BreakingPointType.MEMORY_EXHAUSTION,
                        breaking_value=total_allocated,
                        symptoms=symptoms + ["Safety limit reached"],
                        system_state={"safety_limit": True},
                        recovery_possible=True,
                        error_message="Safety limit prevented host crash"
                    )
                    
                await asyncio.sleep(0.1)
                
        except Exception as e:
            return BreakingPoint(
                test_type=BreakingPointType.MEMORY_EXHAUSTION,
                breaking_value=total_allocated,
                symptoms=symptoms + ["Unexpected exception"],
                system_state={},
                recovery_possible=False,
                error_message=str(e)
            )

class DeadlockCreationTest:
    """Create actual deadlocks that block the system"""
    
    def __init__(self, kernel: EnhancedGAIAKernel):
        self.kernel = kernel
        self.locks = {}
        self.blocked_threads = []
        
    async def create_real_deadlock(self) -> BreakingPoint:
        """Create actual thread deadlocks, not priority inversions"""
        print("🔒 DEADLOCK CREATION - Creating real thread deadlocks")
        
        deadlock_count = 0
        symptoms = []
        
        try:
            # Create circular lock dependencies
            lock_a = threading.Lock()
            lock_b = threading.Lock()
            lock_c = threading.Lock()
            
            self.locks = {'a': lock_a, 'b': lock_b, 'c': lock_c}
            
            async def deadlock_thread_1():
                """Thread 1: A -> B"""
                lock_a.acquire()
                await asyncio.sleep(0.1)  # Give other threads time
                try:
                    if lock_b.acquire(blocking=False):
                        lock_b.release()
                    else:
                        symptoms.append("Thread 1 blocked on B")
                        lock_b.acquire()  # This will deadlock
                        lock_b.release()
                finally:
                    lock_a.release()
                    
            async def deadlock_thread_2():
                """Thread 2: B -> C"""  
                lock_b.acquire()
                await asyncio.sleep(0.1)
                try:
                    if lock_c.acquire(blocking=False):
                        lock_c.release()
                    else:
                        symptoms.append("Thread 2 blocked on C")
                        lock_c.acquire()  # This will deadlock
                        lock_c.release()
                finally:
                    lock_b.release()
                    
            async def deadlock_thread_3():
                """Thread 3: C -> A"""
                lock_c.acquire()
                await asyncio.sleep(0.1)
                try:
                    if lock_a.acquire(blocking=False):
                        lock_a.release()
                    else:
                        symptoms.append("Thread 3 blocked on A")
                        lock_a.acquire()  # This will deadlock
                        lock_a.release()
                finally:
                    lock_c.release()
            
            # Launch deadlock threads
            tasks = [
                asyncio.create_task(deadlock_thread_1()),
                asyncio.create_task(deadlock_thread_2()),
                asyncio.create_task(deadlock_thread_3())
            ]
            
            # Wait for deadlock to form (should timeout)
            try:
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
                # If we get here, no deadlock occurred
                return BreakingPoint(
                    test_type=BreakingPointType.DEADLOCK_CREATION,
                    breaking_value=0,
                    symptoms=["No deadlock created"],
                    system_state={},
                    recovery_possible=True,
                    error_message="Failed to create deadlock"
                )
            except asyncio.TimeoutError:
                # Success! We created a deadlock
                return BreakingPoint(
                    test_type=BreakingPointType.DEADLOCK_CREATION,
                    breaking_value=3,  # 3 threads deadlocked
                    symptoms=symptoms + ["Deadlock confirmed - timeout"],
                    system_state={"deadlocked_threads": 3},
                    recovery_possible=False,
                    error_message="Circular deadlock A->B->C->A"
                )
                
        except Exception as e:
            return BreakingPoint(
                test_type=BreakingPointType.DEADLOCK_CREATION,
                breaking_value=deadlock_count,
                symptoms=symptoms + ["Exception during deadlock"],
                system_state={},
                recovery_possible=True,
                error_message=str(e)
            )

class ThreadLimitTest:
    """Find the actual thread creation limit"""
    
    def __init__(self, kernel: EnhancedGAIAKernel):
        self.kernel = kernel
        self.threads = []
        
    async def find_thread_limit(self) -> BreakingPoint:
        """Keep spawning threads until OS refuses"""
        print("🧵 THREAD LIMIT - Spawning threads until OS breaks")
        
        thread_count = 0
        symptoms = []
        
        def worker_thread():
            """Dummy worker that just sleeps"""
            time.sleep(60)  # Keep thread alive
            
        try:
            while True:
                try:
                    thread = threading.Thread(target=worker_thread)
                    thread.daemon = True
                    thread.start()
                    self.threads.append(thread)
                    thread_count += 1
                    
                    if thread_count % 100 == 0:
                        print(f"  Created {thread_count} threads")
                        
                    # Check for symptoms
                    if thread_count > 1000:
                        symptoms.append("Thread count >1000")
                    if thread_count > 5000:
                        symptoms.append("Thread count >5000") 
                        
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    # We hit the limit!
                    return BreakingPoint(
                        test_type=BreakingPointType.THREAD_LIMIT_BREACH,
                        breaking_value=thread_count,
                        symptoms=symptoms + [f"Thread creation failed: {str(e)[:100]}"],
                        system_state={
                            "active_threads": threading.active_count(),
                            "created_threads": thread_count
                        },
                        recovery_possible=True,
                        error_message=str(e)
                    )
                    
        except KeyboardInterrupt:
            return BreakingPoint(
                test_type=BreakingPointType.THREAD_LIMIT_BREACH,
                breaking_value=thread_count,
                symptoms=symptoms + ["Interrupted by user"],
                system_state={"interrupted": True},
                recovery_possible=True,
                error_message="Test interrupted"
            )

class StackOverflowTest:
    """Force stack overflow through recursion"""
    
    def __init__(self, kernel: EnhancedGAIAKernel):
        self.kernel = kernel
        self.recursion_depth = 0
        
    async def trigger_stack_overflow(self) -> BreakingPoint:
        """Recurse until stack overflow"""
        print("📚 STACK OVERFLOW - Recursing until stack explodes")
        
        try:
            def recursive_bomb(depth: int = 0):
                """Recursively allocate stack frames until overflow"""
                self.recursion_depth = depth
                
                # Allocate some local variables to consume stack space
                local_array = [i for i in range(100)]
                local_string = f"depth_{depth}_" + "x" * 100
                
                if depth % 1000 == 0:
                    print(f"  Recursion depth: {depth}")
                
                # Keep recursing
                return recursive_bomb(depth + 1) + 1
                
            result = recursive_bomb()
            
            # If we get here, no overflow occurred (shouldn't happen)
            return BreakingPoint(
                test_type=BreakingPointType.STACK_OVERFLOW,
                breaking_value=self.recursion_depth,
                symptoms=["No stack overflow - unexpected"],
                system_state={"max_depth": self.recursion_depth},
                recovery_possible=True,
                error_message="Stack overflow didn't occur"
            )
            
        except RecursionError as e:
            return BreakingPoint(
                test_type=BreakingPointType.STACK_OVERFLOW,
                breaking_value=self.recursion_depth,
                symptoms=[f"Stack overflow at depth {self.recursion_depth}"],
                system_state={
                    "max_recursion_depth": self.recursion_depth,
                    "stack_size": resource.getrlimit(resource.RLIMIT_STACK)[0]
                },
                recovery_possible=True,
                error_message=str(e)
            )
        except Exception as e:
            return BreakingPoint(
                test_type=BreakingPointType.STACK_OVERFLOW,
                breaking_value=self.recursion_depth,
                symptoms=[f"Unexpected error at depth {self.recursion_depth}"],
                system_state={},
                recovery_possible=False,
                error_message=str(e)
            )

class BreakingPointTestSuite:
    """Orchestrates all breaking point tests"""
    
    def __init__(self):
        self.kernel = None
        self.breaking_points: List[BreakingPoint] = []
        
    async def run_breaking_point_tests(self) -> List[BreakingPoint]:
        """Run all tests designed to break the system"""
        print("🔥 BREAKING POINT TEST SUITE")
        print("=" * 60)
        print("WARNING: These tests are designed to BREAK the system")
        print("Some tests may cause system instability or crashes")
        print("=" * 60)
        
        # Initialize with minimal memory to make breaking easier
        self.kernel = EnhancedGAIAKernel(memory_size_mb=8)
        await self.kernel.initialize()
        
        test_suite = [
            ("Memory Exhaustion", MemoryExhaustionTest(self.kernel).find_memory_breaking_point),
            ("Stack Overflow", StackOverflowTest(self.kernel).trigger_stack_overflow),
            ("Thread Limit", ThreadLimitTest(self.kernel).find_thread_limit),
            ("Deadlock Creation", DeadlockCreationTest(self.kernel).create_real_deadlock),
        ]
        
        for test_name, test_method in test_suite:
            print(f"\n🚨 RUNNING: {test_name}")
            print("-" * 40)
            
            try:
                breaking_point = await test_method()
                self.breaking_points.append(breaking_point)
                
                self._print_breaking_point(breaking_point)
                
                # Clean up between tests
                if hasattr(self, 'memory_bombs'):
                    self.memory_bombs.clear()
                    
                # Brief recovery time
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"❌ {test_name} failed to complete: {e}")
                error_bp = BreakingPoint(
                    test_type=BreakingPointType.MEMORY_EXHAUSTION,
                    breaking_value=0,
                    symptoms=[f"Test framework error: {str(e)[:100]}"],
                    system_state={},
                    recovery_possible=False,
                    error_message=str(e)
                )
                self.breaking_points.append(error_bp)
        
        await self._cleanup()
        return self.breaking_points
    
    def _print_breaking_point(self, bp: BreakingPoint):
        """Print breaking point details"""
        print(f"💥 BREAKING POINT FOUND:")
        print(f"   Type: {bp.test_type.value}")
        print(f"   Breaking Value: {bp.breaking_value}")
        print(f"   Recovery Possible: {'✅' if bp.recovery_possible else '❌'}")
        print(f"   Symptoms: {', '.join(bp.symptoms)}")
        print(f"   Error: {bp.error_message[:100]}...")
        
    async def _cleanup(self):
        """Attempt cleanup after destructive tests"""
        print("\n🧹 Attempting cleanup...")
        try:
            if self.kernel:
                await self.kernel.cleanup()
        except:
            pass
            
    def generate_breaking_point_report(self) -> str:
        """Generate comprehensive report of all breaking points"""
        report = ["🔥 BREAKING POINT ANALYSIS REPORT", "=" * 60]
        
        if not self.breaking_points:
            report.append("❌ No breaking points found - tests may be too weak")
            return "\n".join(report)
        
        report.append(f"📊 SUMMARY:")
        report.append(f"   Total Breaking Points Found: {len(self.breaking_points)}")
        report.append(f"   Recoverable Failures: {sum(1 for bp in self.breaking_points if bp.recovery_possible)}")
        report.append(f"   Critical Failures: {sum(1 for bp in self.breaking_points if not bp.recovery_possible)}")
        
        report.append(f"\n📋 DETAILED BREAKDOWN:")
        for i, bp in enumerate(self.breaking_points, 1):
            report.append(f"\n{i}. {bp.test_type.value.upper()}")
            report.append(f"   Breaking Point: {bp.breaking_value}")
            report.append(f"   Symptoms: {'; '.join(bp.symptoms)}")
            report.append(f"   Recovery: {'Possible' if bp.recovery_possible else 'CRITICAL - System unstable'}")
            report.append(f"   Details: {bp.error_message[:150]}...")
        
        report.append(f"\n🏆 SYSTEM RESILIENCE ASSESSMENT:")
        critical_failures = [bp for bp in self.breaking_points if not bp.recovery_possible]
        
        if len(critical_failures) == 0:
            report.append("🟢 EXCELLENT - System fails gracefully with recovery")
        elif len(critical_failures) <= len(self.breaking_points) * 0.3:
            report.append("🟡 GOOD - Most failures are recoverable")
        else:
            report.append("🔴 POOR - Multiple critical failures detected")
            
        report.append(f"\n💡 RECOMMENDATIONS:")
        if len(critical_failures) > 0:
            report.append("   🔧 Address critical failure modes before production")
            report.append("   🔧 Implement better error handling and recovery")
        else:
            report.append("   ✅ System demonstrates good failure resilience")
            report.append("   ✅ Breaking points are within acceptable parameters")
        
        return "\n".join(report)

async def run_breaking_point_tests():
    """Main entry point for breaking point tests"""
    suite = BreakingPointTestSuite()
    
    print("⚠️  WARNING: DESTRUCTIVE TESTING AHEAD")
    print("These tests will attempt to break the system")
    print("Press Ctrl+C within 5 seconds to abort...")
    
    try:
        await asyncio.sleep(5)
    except KeyboardInterrupt:
        print("\n❌ Tests aborted by user")
        return
    
    results = await suite.run_breaking_point_tests()
    
    print(suite.generate_breaking_point_report())
    
    return results

if __name__ == "__main__":
    asyncio.run(run_breaking_point_tests())