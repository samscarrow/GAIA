"""
Test harness for Hierarchical Memory Management System
Simulates realistic load patterns and measures performance
"""

import asyncio
import time
import random
import numpy as np
from typing import List, Dict, Any
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.hierarchical_memory import HierarchicalMemoryManager, ZoneState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoadSimulator:
    """Simulates realistic memory access patterns"""
    
    def __init__(self, memory_manager: HierarchicalMemoryManager):
        self.memory = memory_manager
        self.stored_keys: List[str] = []
        self.access_patterns = {
            'sequential': 0.2,    # Sequential access
            'random': 0.3,         # Random access
            'locality': 0.3,       # Temporal locality
            'burst': 0.2          # Burst access
        }
        
        # Performance metrics
        self.operation_times: List[float] = []
        self.successful_ops = 0
        self.failed_ops = 0
        
    async def generate_workload(self, duration_seconds: int = 60, 
                               operations_per_second: int = 100):
        """Generate synthetic workload"""
        logger.info(f"Starting workload simulation for {duration_seconds}s at {operations_per_second} ops/s")
        
        start_time = time.time()
        operation_count = 0
        
        while time.time() - start_time < duration_seconds:
            # Determine access pattern for this batch
            pattern = self._select_pattern()
            
            # Execute batch of operations
            batch_size = random.randint(1, 10)
            await self._execute_batch(pattern, batch_size)
            
            operation_count += batch_size
            
            # Control rate
            await asyncio.sleep(batch_size / operations_per_second)
            
            # Log progress
            if operation_count % 100 == 0:
                elapsed = time.time() - start_time
                actual_rate = operation_count / elapsed
                logger.info(f"Progress: {operation_count} operations, "
                          f"rate: {actual_rate:.1f} ops/s")
        
        return self._get_metrics()
    
    def _select_pattern(self) -> str:
        """Select access pattern based on probabilities"""
        rand = random.random()
        cumulative = 0
        for pattern, prob in self.access_patterns.items():
            cumulative += prob
            if rand < cumulative:
                return pattern
        return 'random'
    
    async def _execute_batch(self, pattern: str, batch_size: int):
        """Execute a batch of operations based on pattern"""
        operations = []
        
        if pattern == 'sequential':
            # Sequential writes followed by sequential reads
            base_key = f"seq_{time.time()}"
            for i in range(batch_size):
                operations.append(self._store_operation(f"{base_key}_{i}"))
        
        elif pattern == 'random':
            # Mix of random reads and writes
            for _ in range(batch_size):
                if random.random() < 0.7 and self.stored_keys:
                    # 70% reads
                    operations.append(self._retrieve_operation(random.choice(self.stored_keys)))
                else:
                    # 30% writes
                    operations.append(self._store_operation(f"random_{time.time()}_{random.randint(0, 1000)}"))
        
        elif pattern == 'locality':
            # Access recently used items
            if self.stored_keys:
                recent_keys = self.stored_keys[-min(20, len(self.stored_keys)):]
                for _ in range(batch_size):
                    operations.append(self._retrieve_operation(random.choice(recent_keys)))
            else:
                # Fallback to writes if no keys stored yet
                for i in range(batch_size):
                    operations.append(self._store_operation(f"locality_{time.time()}_{i}"))
        
        elif pattern == 'burst':
            # Rapid access to same items
            if self.stored_keys:
                target_key = random.choice(self.stored_keys)
                for _ in range(batch_size):
                    operations.append(self._retrieve_operation(target_key))
            else:
                # Burst writes
                base_key = f"burst_{time.time()}"
                for i in range(batch_size):
                    operations.append(self._store_operation(f"{base_key}_{i}"))
        
        # Execute all operations concurrently
        results = await asyncio.gather(*operations, return_exceptions=True)
        
        # Track results
        for result in results:
            if isinstance(result, Exception):
                self.failed_ops += 1
                logger.error(f"Operation failed: {result}")
            else:
                self.successful_ops += 1
    
    async def _store_operation(self, key: str):
        """Execute a store operation"""
        start = time.time()
        
        # Generate synthetic embedding
        embedding = np.random.randn(512).astype(np.float32)  # 2KB embedding
        
        # Generate associations
        associations = set()
        if self.stored_keys and random.random() < 0.3:
            # 30% chance of associations
            num_associations = min(random.randint(1, 5), len(self.stored_keys))
            associations = set(random.sample(self.stored_keys, num_associations))
        
        # Store
        zone_id = await self.memory.store(
            key=key,
            embedding=embedding,
            associations=associations,
            semantic_category=self._generate_category()
        )
        
        self.stored_keys.append(key)
        elapsed = time.time() - start
        self.operation_times.append(elapsed)
        
        return zone_id
    
    async def _retrieve_operation(self, key: str):
        """Execute a retrieve operation"""
        start = time.time()
        
        result = await self.memory.retrieve(key)
        
        elapsed = time.time() - start
        self.operation_times.append(elapsed)
        
        return result
    
    def _generate_category(self) -> str:
        """Generate semantic category"""
        categories = ['vision', 'language', 'reasoning', 'memory', 'planning']
        return random.choice(categories)
    
    def _get_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if not self.operation_times:
            return {}
        
        return {
            'total_operations': self.successful_ops + self.failed_ops,
            'successful_operations': self.successful_ops,
            'failed_operations': self.failed_ops,
            'success_rate': self.successful_ops / (self.successful_ops + self.failed_ops + 1),
            'avg_latency_ms': np.mean(self.operation_times) * 1000,
            'p50_latency_ms': np.percentile(self.operation_times, 50) * 1000,
            'p95_latency_ms': np.percentile(self.operation_times, 95) * 1000,
            'p99_latency_ms': np.percentile(self.operation_times, 99) * 1000,
            'throughput_ops_per_sec': len(self.operation_times) / sum(self.operation_times)
        }

class StressTest:
    """Stress testing for memory limits"""
    
    def __init__(self, memory_manager: HierarchicalMemoryManager):
        self.memory = memory_manager
    
    async def test_memory_pressure(self):
        """Test behavior under memory pressure"""
        logger.info("Starting memory pressure test...")
        
        stored_keys = []
        metrics = {
            'zones_created': [],
            'zones_frozen': [],
            'memory_used': [],
            'compression_ratio': []
        }
        
        # Fill memory to 150% capacity (should trigger compression)
        target_operations = 1000
        for i in range(target_operations):
            key = f"pressure_test_{i}"
            embedding = np.random.randn(1024).astype(np.float32)  # 4KB each
            
            await self.memory.store(key, embedding)
            stored_keys.append(key)
            
            if i % 100 == 0:
                status = self.memory.get_status()
                metrics['zones_created'].append(status['total_zones'])
                metrics['zones_frozen'].append(status['frozen_zones'])
                metrics['memory_used'].append(status['memory_used_mb'])
                
                logger.info(f"Iteration {i}: {status['total_zones']} zones, "
                          f"{status['frozen_zones']} frozen, "
                          f"{status['memory_used_mb']:.1f}MB used")
        
        # Test retrieval of compressed data
        logger.info("Testing retrieval of compressed data...")
        retrieval_times = []
        for _ in range(100):
            key = random.choice(stored_keys)
            start = time.time()
            result = await self.memory.retrieve(key)
            retrieval_times.append(time.time() - start)
            
            if result is None:
                logger.error(f"Failed to retrieve {key}")
        
        return {
            'final_zones': metrics['zones_created'][-1],
            'frozen_zones': metrics['zones_frozen'][-1],
            'peak_memory_mb': max(metrics['memory_used']),
            'avg_retrieval_ms': np.mean(retrieval_times) * 1000,
            'compression_triggered': any(metrics['zones_frozen'])
        }
    
    async def test_concurrent_access(self):
        """Test concurrent read/write operations"""
        logger.info("Starting concurrent access test...")
        
        # Pre-populate some data
        keys = []
        for i in range(100):
            key = f"concurrent_{i}"
            embedding = np.random.randn(512).astype(np.float32)
            await self.memory.store(key, embedding)
            keys.append(key)
        
        # Concurrent operations
        async def reader_task():
            for _ in range(50):
                key = random.choice(keys)
                await self.memory.retrieve(key)
                await asyncio.sleep(0.01)
        
        async def writer_task():
            for i in range(50):
                key = f"concurrent_new_{i}"
                embedding = np.random.randn(512).astype(np.float32)
                await self.memory.store(key, embedding)
                await asyncio.sleep(0.01)
        
        # Run concurrent tasks
        start = time.time()
        await asyncio.gather(
            *[reader_task() for _ in range(5)],
            *[writer_task() for _ in range(5)]
        )
        elapsed = time.time() - start
        
        return {
            'concurrent_duration_seconds': elapsed,
            'operations_per_second': 500 / elapsed,  # 10 tasks * 50 ops each
            'final_status': self.memory.get_status()
        }

async def run_comprehensive_tests():
    """Run all tests and generate report"""
    logger.info("=" * 60)
    logger.info("HIERARCHICAL MEMORY MANAGEMENT SYSTEM TEST SUITE")
    logger.info("=" * 60)
    
    # Initialize memory manager
    memory = HierarchicalMemoryManager(
        total_memory_mb=256,  # 256MB for testing
        compression_threshold=0.7
    )
    await memory.initialize()
    
    results = {}
    
    # Test 1: Load simulation
    logger.info("\n--- TEST 1: Load Simulation ---")
    simulator = LoadSimulator(memory)
    load_metrics = await simulator.generate_workload(
        duration_seconds=30,
        operations_per_second=50
    )
    results['load_simulation'] = load_metrics
    
    # Test 2: Memory pressure
    logger.info("\n--- TEST 2: Memory Pressure ---")
    stress = StressTest(memory)
    pressure_metrics = await stress.test_memory_pressure()
    results['memory_pressure'] = pressure_metrics
    
    # Test 3: Concurrent access
    logger.info("\n--- TEST 3: Concurrent Access ---")
    concurrent_metrics = await stress.test_concurrent_access()
    results['concurrent_access'] = concurrent_metrics
    
    # Final status
    final_status = memory.get_status()
    results['final_status'] = final_status
    
    # Generate report
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    logger.info("\nLoad Simulation Results:")
    for key, value in load_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")
    
    logger.info("\nMemory Pressure Results:")
    for key, value in pressure_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")
    
    logger.info("\nConcurrent Access Results:")
    logger.info(f"  Duration: {concurrent_metrics['concurrent_duration_seconds']:.2f}s")
    logger.info(f"  Throughput: {concurrent_metrics['operations_per_second']:.2f} ops/s")
    
    logger.info("\nFinal System Status:")
    for key, value in final_status.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Performance verdict
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE VERDICT")
    logger.info("=" * 60)
    
    if load_metrics.get('p95_latency_ms', float('inf')) < 50:
        logger.info("✅ Latency: EXCELLENT (<50ms p95)")
    elif load_metrics.get('p95_latency_ms', float('inf')) < 100:
        logger.info("⚠️  Latency: ACCEPTABLE (<100ms p95)")
    else:
        logger.info("❌ Latency: NEEDS OPTIMIZATION (>100ms p95)")
    
    if pressure_metrics.get('compression_triggered', False):
        logger.info("✅ Memory Management: WORKING (compression triggered)")
    else:
        logger.info("❌ Memory Management: NOT TRIGGERED")
    
    if final_status.get('cache_hit_rate', 0) > 0.8:
        logger.info("✅ Cache Performance: EXCELLENT (>80% hit rate)")
    elif final_status.get('cache_hit_rate', 0) > 0.6:
        logger.info("⚠️  Cache Performance: ACCEPTABLE (>60% hit rate)")
    else:
        logger.info("❌ Cache Performance: POOR (<60% hit rate)")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_comprehensive_tests())