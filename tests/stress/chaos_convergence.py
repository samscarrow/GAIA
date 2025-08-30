"""
Chaos Convergence Stress Test - The Ultimate GAIA Architecture Breaking Test
Combines the most brutal suggestions from our LM Studio colleagues to find breaking points.
"""

import asyncio
import threading
import time
import random
import gc
import psutil
import os
import signal
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kernel.enhanced_core import EnhancedGAIAKernel
from memory.hierarchical_memory import HierarchicalMemoryManager, ZoneState

logging.basicConfig(level=logging.ERROR)  # Reduce noise during chaos

class ChaosLevel(Enum):
    """Chaos intensity levels"""
    MILD_CHAOS = "mild"
    MODERATE_CHAOS = "moderate" 
    SEVERE_CHAOS = "severe"
    APOCALYPTIC_CHAOS = "apocalyptic"

class AttackVector(Enum):
    """Different attack vectors"""
    MEMORY_FRAGMENTATION_BOMB = "memory_frag"
    PRIORITY_INVERSION_CASCADE = "priority_inversion"
    CONCURRENT_OVERLOAD = "concurrent_overload"
    FAILURE_CASCADE = "failure_cascade"
    RESOURCE_STARVATION = "resource_starvation"
    SEMANTIC_COLLISION = "semantic_collision"

@dataclass
class ChaosMetrics:
    """Metrics tracked during chaos testing"""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # System metrics
    peak_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    total_thoughts_spawned: int = 0
    successful_thoughts: int = 0
    failed_thoughts: int = 0
    crashed_thoughts: int = 0
    
    # Memory system metrics
    zones_created: int = 0
    zones_corrupted: int = 0
    compression_failures: int = 0
    btree_fragmentation_percent: float = 0.0
    checkpoint_failures: int = 0
    
    # Priority system metrics
    priority_inversions_detected: int = 0
    deadlocks_detected: int = 0
    preemption_failures: int = 0
    
    # Resource metrics
    resource_exhaustions: int = 0
    fallback_activations: int = 0
    system_recoveries: int = 0
    
    def duration(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

class MemoryFragmentationBomb:
    """Creates pathological memory fragmentation patterns"""
    
    def __init__(self, kernel: EnhancedGAIAKernel):
        self.kernel = kernel
        self.fragmentation_data: List[bytes] = []
        self.semantic_collision_keys: List[str] = []
    
    async def create_semantic_collision_storm(self, collision_count: int = 10000):
        """Create identical semantic signatures to trigger B-tree pathology"""
        print(f"💣 Deploying semantic collision storm ({collision_count} collisions)")
        
        # Phase 1: Create identical semantic zones
        base_key = "semantic_collision"
        identical_embedding = np.random.randn(512).astype(np.float32)
        
        for i in range(collision_count):
            key = f"{base_key}_{i}"
            self.semantic_collision_keys.append(key)
            
            try:
                # All have identical semantic signature - should merge but won't due to different keys
                await self.kernel.memory_manager.store(
                    key, identical_embedding, semantic_category="collision_category"
                )
                
                # Every 1000th item, trigger a random access to fragment B-tree
                if i % 1000 == 0 and i > 0:
                    random_key = random.choice(self.semantic_collision_keys[:i])
                    await self.kernel.memory_manager.retrieve(random_key)
                
                # Force memory pressure every 2000 items
                if i % 2000 == 0:
                    await self.kernel.memory_manager._manage_memory_pressure()
                
            except Exception as e:
                print(f"    💀 Collision {i} failed: {e}")
                break
        
        return len(self.semantic_collision_keys)
    
    async def interleaved_fragmentation_pattern(self, pattern_size: int = 5000):
        """Create interleaved allocation/deallocation to maximize fragmentation"""
        print(f"🔀 Creating interleaved fragmentation pattern ({pattern_size} operations)")
        
        allocated_keys = []
        
        for i in range(pattern_size):
            try:
                if random.random() < 0.7:  # 70% allocate, 30% deallocate
                    # Allocate with random size
                    size = random.randint(100, 2000)
                    embedding = np.random.randn(size).astype(np.float32)
                    key = f"frag_{i}_{size}"
                    
                    await self.kernel.memory_manager.store(
                        key, embedding, semantic_category=f"frag_cat_{i % 20}"
                    )
                    allocated_keys.append(key)
                    
                else:
                    # Deallocate random existing key
                    if allocated_keys:
                        # Simulate deallocation by accessing and letting GC handle it
                        random_key = random.choice(allocated_keys)
                        try:
                            await self.kernel.memory_manager.retrieve(random_key)
                        except:
                            pass
                        allocated_keys.remove(random_key)
            except Exception as e:
                print(f"    💥 Fragmentation operation {i} failed: {e}")
        
        return len(allocated_keys)

class PriorityInversionCascade:
    """Creates priority inversion scenarios that cascade into deadlocks"""
    
    def __init__(self, kernel: EnhancedGAIAKernel):
        self.kernel = kernel
        self.priority_locks: Dict[int, threading.Lock] = {}
        self.blocked_tasks: List[str] = []
    
    async def create_priority_deadlock_chain(self, chain_length: int = 100):
        """Create a chain of priority dependencies that deadlock"""
        print(f"🔗 Creating priority deadlock chain (length: {chain_length})")
        
        # Create a chain: Priority 100 waits for 99, 99 waits for 98, ..., 1 waits for 100
        thought_ids = []
        
        try:
            # Setup models that will create the deadlock
            for i in range(min(chain_length, 20)):  # Limit model count for sanity
                model_id = f"deadlock_model_{i}"
                self.kernel.register_model_with_fallback(
                    model_id, "deadlock", 100, 200
                )
            
            # Create the deadlock chain
            for priority in range(chain_length, 0, -1):
                model_id = f"deadlock_model_{priority % 20}"
                
                # Each thought waits for the next priority level
                dependent_priority = priority - 1 if priority > 1 else chain_length
                
                thought_id = await self.kernel.spawn_thought_with_priority(
                    model_id,
                    {
                        "priority": priority,
                        "depends_on": dependent_priority,
                        "deadlock_chain": True
                    },
                    priority=priority
                )
                thought_ids.append(thought_id)
                
                # Small delay to ensure ordering
                await asyncio.sleep(0.001)
        
        except Exception as e:
            print(f"    ⚠️ Deadlock creation failed: {e}")
        
        return thought_ids
    
    async def priority_starvation_attack(self, starved_priority: int = 5, 
                                        attack_duration: float = 30.0):
        """Starve a specific priority level by flooding higher priorities"""
        print(f"🥶 Priority starvation attack on level {starved_priority} for {attack_duration}s")
        
        start_time = time.time()
        starved_thoughts = []
        flooding_thoughts = []
        
        try:
            # Create the victim (low priority task)
            victim_thought = await self.kernel.spawn_thought_with_priority(
                "deadlock_model_0",
                {"victim": True, "priority": starved_priority},
                priority=starved_priority
            )
            starved_thoughts.append(victim_thought)
            
            # Flood with higher priority tasks
            while time.time() - start_time < attack_duration:
                flood_priority = starved_priority + random.randint(10, 50)
                
                flood_thought = await self.kernel.spawn_thought_with_priority(
                    "deadlock_model_1",
                    {"flood_attack": True, "priority": flood_priority},
                    priority=flood_priority
                )
                flooding_thoughts.append(flood_thought)
                
                # Rapid fire
                await asyncio.sleep(0.01)
        
        except Exception as e:
            print(f"    💀 Starvation attack failed: {e}")
        
        return starved_thoughts, flooding_thoughts

class FailureCascadeSimulator:
    """Simulates cascading failures that propagate through the system"""
    
    def __init__(self, kernel: EnhancedGAIAKernel):
        self.kernel = kernel
        self.injected_failures = 0
        self.cascade_depth = 0
    
    async def memory_corruption_cascade(self):
        """Simulate memory corruption that cascades through the system"""
        print("🦠 Initiating memory corruption cascade")
        
        try:
            # Force all models to fail simultaneously
            for model in self.kernel.models.values():
                model.current_accuracy = 0.01  # Near-zero accuracy
                model.failure_count = 100
                self.injected_failures += 1
            
            # Corrupt memory zones
            for zone in self.kernel.memory_manager.zones.values():
                if random.random() < 0.3:  # 30% corruption rate
                    zone.state = ZoneState.FROZEN  # Force thaw operations
                    zone.compressed_data = b"corrupted"
                    self.injected_failures += 1
            
            # Force fallback cascade
            cascade_thoughts = []
            for i in range(50):
                try:
                    thought_id = await self.kernel.spawn_thought_with_priority(
                        list(self.kernel.models.keys())[0],
                        {"cascade_test": i},
                        priority=random.randint(1, 100)
                    )
                    cascade_thoughts.append(thought_id)
                except Exception as e:
                    self.cascade_depth += 1
                    print(f"    🌊 Cascade depth: {self.cascade_depth}")
            
            return cascade_thoughts
            
        except Exception as e:
            print(f"    💥 Cascade simulation failed: {e}")
            return []

class ConcurrentOverloadAttack:
    """Overwhelms the system with concurrent operations"""
    
    def __init__(self, kernel: EnhancedGAIAKernel):
        self.kernel = kernel
        self.concurrent_tasks = []
    
    async def thread_explosion(self, explosion_factor: int = 1000):
        """Create thread explosion beyond system limits"""
        print(f"💥 Thread explosion attack (factor: {explosion_factor})")
        
        async def chaos_worker(worker_id: int):
            """Worker that creates more chaos"""
            try:
                # Each worker spawns multiple thoughts
                for i in range(10):
                    thought_id = await self.kernel.spawn_thought_with_priority(
                        "deadlock_model_0",
                        {"chaos_worker": worker_id, "sub_task": i},
                        priority=random.randint(1, 100)
                    )
                
                # Worker also creates memory pressure
                chaos_data = np.random.randn(random.randint(1000, 10000)).astype(np.float32)
                await self.kernel.memory_manager.store(
                    f"chaos_{worker_id}",
                    chaos_data,
                    semantic_category=f"chaos_cat_{worker_id % 50}"
                )
                
                return worker_id
                
            except Exception as e:
                return f"worker_{worker_id}_failed: {e}"
        
        # Create thread explosion
        tasks = []
        for worker_id in range(explosion_factor):
            task = asyncio.create_task(chaos_worker(worker_id))
            tasks.append(task)
        
        # Don't wait for all - let them run wild
        return tasks

class ChaosConvergenceTest:
    """The ultimate GAIA stress test combining all attack vectors"""
    
    def __init__(self, chaos_level: ChaosLevel = ChaosLevel.SEVERE_CHAOS):
        self.chaos_level = chaos_level
        self.metrics = ChaosMetrics()
        self.kernel: Optional[EnhancedGAIAKernel] = None
        
        # Attack components
        self.memory_bomb: Optional[MemoryFragmentationBomb] = None
        self.priority_cascade: Optional[PriorityInversionCascade] = None
        self.failure_simulator: Optional[FailureCascadeSimulator] = None
        self.overload_attack: Optional[ConcurrentOverloadAttack] = None
        
        # Monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.system_crashed = False
    
    async def initialize_chaos_environment(self):
        """Setup the environment for maximum chaos"""
        print("🎭 Initializing Chaos Environment...")
        
        # Smaller memory to trigger pressure faster
        memory_size = {
            ChaosLevel.MILD_CHAOS: 128,
            ChaosLevel.MODERATE_CHAOS: 64,
            ChaosLevel.SEVERE_CHAOS: 32,
            ChaosLevel.APOCALYPTIC_CHAOS: 16
        }[self.chaos_level]
        
        self.kernel = EnhancedGAIAKernel(memory_size_mb=memory_size)
        await self.kernel.initialize()
        
        # Initialize attack components
        self.memory_bomb = MemoryFragmentationBomb(self.kernel)
        self.priority_cascade = PriorityInversionCascade(self.kernel)
        self.failure_simulator = FailureCascadeSimulator(self.kernel)
        self.overload_attack = ConcurrentOverloadAttack(self.kernel)
        
        # Setup models for chaos
        for i in range(10):
            self.kernel.register_model_with_fallback(
                f"chaos_model_{i}", f"chaos_type_{i}", 
                memory_footprint=random.randint(100, 500),
                vram_required=random.randint(500, 2000),
                fallback_model_id=f"chaos_model_{(i+1) % 10}"  # Circular fallbacks
            )
        
        # Start system monitoring
        self.monitoring_task = asyncio.create_task(self._monitor_system())
        
        print(f"🔥 Chaos environment initialized with {memory_size}MB memory")
        print(f"💀 Chaos level: {self.chaos_level.value}")
    
    async def execute_chaos_convergence(self, duration_seconds: int = 120):
        """Execute the full chaos convergence attack"""
        print(f"\n🌪️ EXECUTING CHAOS CONVERGENCE")
        print(f"Duration: {duration_seconds} seconds")
        print(f"Expected outcome: Total system breakdown")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Phase 1: Memory Fragmentation Bomb (0-30s)
            print(f"\n⚡ PHASE 1: Memory Fragmentation Bomb")
            fragmentation_task = asyncio.create_task(
                self.memory_bomb.create_semantic_collision_storm(5000)
            )
            interleave_task = asyncio.create_task(
                self.memory_bomb.interleaved_fragmentation_pattern(2000)
            )
            
            await asyncio.sleep(10)  # Let fragmentation build
            
            # Phase 2: Priority Cascade (10-60s) 
            print(f"\n⚡ PHASE 2: Priority Inversion Cascade")
            deadlock_task = asyncio.create_task(
                self.priority_cascade.create_priority_deadlock_chain(50)
            )
            starvation_task = asyncio.create_task(
                self.priority_cascade.priority_starvation_attack(5, 30)
            )
            
            await asyncio.sleep(10)
            
            # Phase 3: Concurrent Overload (20-90s)
            print(f"\n⚡ PHASE 3: Thread Explosion")
            explosion_factor = {
                ChaosLevel.MILD_CHAOS: 100,
                ChaosLevel.MODERATE_CHAOS: 300,
                ChaosLevel.SEVERE_CHAOS: 500,
                ChaosLevel.APOCALYPTIC_CHAOS: 1000
            }[self.chaos_level]
            
            explosion_tasks = await self.overload_attack.thread_explosion(explosion_factor)
            
            await asyncio.sleep(20)
            
            # Phase 4: Failure Cascade (40-120s)
            print(f"\n⚡ PHASE 4: Cascading System Failure")
            cascade_task = asyncio.create_task(
                self.failure_simulator.memory_corruption_cascade()
            )
            
            # Let chaos run until time limit or system death
            end_time = start_time + duration_seconds
            while time.time() < end_time and not self.system_crashed:
                await asyncio.sleep(1)
                
                # Check if system is still responsive
                try:
                    status = self.kernel.get_enhanced_status()
                    if status['kernel']['failed_thoughts'] > status['kernel']['total_thoughts_executed'] * 0.8:
                        print("💀 System failure rate >80% - declaring system death")
                        self.system_crashed = True
                        break
                except Exception:
                    print("💀 System unresponsive - declaring system death")
                    self.system_crashed = True
                    break
                    
        except Exception as e:
            print(f"💥 Chaos convergence exception: {e}")
            self.system_crashed = True
        
        self.metrics.end_time = time.time()
        await self._collect_final_metrics()
    
    async def _monitor_system(self):
        """Monitor system metrics during chaos"""
        while not self.system_crashed:
            try:
                # CPU and memory
                process = psutil.Process()
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                
                self.metrics.peak_cpu_percent = max(self.metrics.peak_cpu_percent, cpu_percent)
                self.metrics.peak_memory_mb = max(self.metrics.peak_memory_mb, memory_mb)
                
                # System status
                if self.kernel:
                    status = self.kernel.get_enhanced_status()
                    self.metrics.total_thoughts_spawned = status['kernel']['total_thoughts_executed']
                    self.metrics.failed_thoughts = status['kernel']['failed_thoughts']
                    self.metrics.zones_created = status['memory']['total_zones']
                
                await asyncio.sleep(1)
                
            except Exception:
                break
    
    async def _collect_final_metrics(self):
        """Collect final metrics after chaos"""
        try:
            if self.kernel:
                status = self.kernel.get_enhanced_status()
                
                self.metrics.total_thoughts_spawned = status['kernel']['total_thoughts_executed']
                self.metrics.successful_thoughts = self.metrics.total_thoughts_spawned - status['kernel']['failed_thoughts']
                self.metrics.failed_thoughts = status['kernel']['failed_thoughts']
                self.metrics.zones_created = status['memory']['total_zones']
                self.metrics.zones_corrupted = status['memory']['frozen_zones']  # Approximation
                
                # Calculate fragmentation
                if len(self.kernel.memory_manager.zones) > 0:
                    total_zones = len(self.kernel.memory_manager.zones)
                    frozen_zones = sum(1 for z in self.kernel.memory_manager.zones.values() 
                                     if z.state == ZoneState.FROZEN)
                    self.metrics.btree_fragmentation_percent = (frozen_zones / total_zones) * 100
                
        except Exception as e:
            print(f"⚠️ Error collecting final metrics: {e}")
    
    def generate_chaos_report(self) -> Dict[str, Any]:
        """Generate comprehensive chaos test report"""
        duration = self.metrics.duration()
        
        report = {
            "chaos_level": self.chaos_level.value,
            "system_survived": not self.system_crashed,
            "duration_seconds": duration,
            "metrics": {
                "peak_memory_mb": self.metrics.peak_memory_mb,
                "peak_cpu_percent": self.metrics.peak_cpu_percent,
                "total_thoughts_spawned": self.metrics.total_thoughts_spawned,
                "successful_thoughts": self.metrics.successful_thoughts,
                "failed_thoughts": self.metrics.failed_thoughts,
                "success_rate": (self.metrics.successful_thoughts / max(self.metrics.total_thoughts_spawned, 1)),
                "zones_created": self.metrics.zones_created,
                "btree_fragmentation_percent": self.metrics.btree_fragmentation_percent,
            },
            "attack_analysis": {
                "memory_fragmentation": {
                    "semantic_collisions": len(self.memory_bomb.semantic_collision_keys) if self.memory_bomb else 0,
                    "fragmentation_level": self.metrics.btree_fragmentation_percent
                },
                "priority_system": {
                    "inversions_detected": self.metrics.priority_inversions_detected,
                    "deadlocks_detected": self.metrics.deadlocks_detected
                },
                "failure_cascade": {
                    "injected_failures": self.failure_simulator.injected_failures if self.failure_simulator else 0,
                    "cascade_depth": self.failure_simulator.cascade_depth if self.failure_simulator else 0
                }
            }
        }
        
        return report
    
    def print_chaos_report(self, report: Dict[str, Any]):
        """Print chaos test results"""
        print(f"\n" + "=" * 60)
        print(f"🌪️ CHAOS CONVERGENCE RESULTS")
        print(f"=" * 60)
        
        print(f"Chaos Level: {report['chaos_level'].upper()}")
        print(f"Duration: {report['duration_seconds']:.2f}s")
        print(f"System Survived: {'🟢 YES' if report['system_survived'] else '🔴 NO'}")
        
        metrics = report['metrics']
        print(f"\n📊 SYSTEM METRICS:")
        print(f"  Peak Memory: {metrics['peak_memory_mb']:.1f}MB")
        print(f"  Peak CPU: {metrics['peak_cpu_percent']:.1f}%")
        print(f"  Thoughts Spawned: {metrics['total_thoughts_spawned']}")
        print(f"  Success Rate: {metrics['success_rate']:.1%}")
        print(f"  Memory Zones: {metrics['zones_created']}")
        print(f"  B-Tree Fragmentation: {metrics['btree_fragmentation_percent']:.1f}%")
        
        attacks = report['attack_analysis']
        print(f"\n💣 ATTACK ANALYSIS:")
        print(f"  Semantic Collisions: {attacks['memory_fragmentation']['semantic_collisions']}")
        print(f"  Fragmentation Level: {attacks['memory_fragmentation']['fragmentation_level']:.1f}%")
        print(f"  Priority Inversions: {attacks['priority_system']['inversions_detected']}")
        print(f"  Cascade Depth: {attacks['failure_cascade']['cascade_depth']}")
        
        print(f"\n🏆 FINAL VERDICT:")
        if not report['system_survived']:
            print("💀 SYSTEM DESTROYED - Chaos convergence successful")
            print("   The architecture has critical weaknesses under extreme load")
        elif metrics['success_rate'] < 0.5:
            print("🟡 SYSTEM DEGRADED - Partial chaos success")
            print("   System survived but with significant performance issues")
        elif metrics['btree_fragmentation_percent'] > 70:
            print("🟠 MEMORY CORRUPTED - Fragmentation bomb effective")
            print("   Memory system compromised but kernel survived")
        else:
            print("🟢 SYSTEM RESILIENT - Chaos convergence failed")
            print("   Architecture withstood extreme stress conditions")
    
    async def cleanup(self):
        """Cleanup after chaos test"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        if self.kernel and self.kernel.memory_manager and self.kernel.memory_manager.maintenance_task:
            self.kernel.memory_manager.maintenance_task.cancel()
            try:
                await self.kernel.memory_manager.maintenance_task
            except asyncio.CancelledError:
                pass

async def run_chaos_convergence(chaos_level: ChaosLevel = ChaosLevel.SEVERE_CHAOS, 
                               duration: int = 60):
    """Run the ultimate chaos convergence test"""
    chaos_test = ChaosConvergenceTest(chaos_level)
    
    try:
        await chaos_test.initialize_chaos_environment()
        await chaos_test.execute_chaos_convergence(duration)
        
        report = chaos_test.generate_chaos_report()
        chaos_test.print_chaos_report(report)
        
        return report
        
    finally:
        await chaos_test.cleanup()

if __name__ == "__main__":
    import sys
    
    # Parse chaos level from command line
    level_map = {
        "mild": ChaosLevel.MILD_CHAOS,
        "moderate": ChaosLevel.MODERATE_CHAOS,
        "severe": ChaosLevel.SEVERE_CHAOS,
        "apocalyptic": ChaosLevel.APOCALYPTIC_CHAOS
    }
    
    chaos_level = ChaosLevel.SEVERE_CHAOS
    if len(sys.argv) > 1 and sys.argv[1] in level_map:
        chaos_level = level_map[sys.argv[1]]
    
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    
    print(f"🌪️ Starting Chaos Convergence Test")
    print(f"Level: {chaos_level.value}")
    print(f"Duration: {duration}s")
    print(f"⚠️ WARNING: This test is designed to break the system!")
    
    asyncio.run(run_chaos_convergence(chaos_level, duration))