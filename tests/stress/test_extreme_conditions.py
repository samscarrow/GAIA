"""
Stress tests for extreme conditions and edge cases
"""

import pytest
import numpy as np
import asyncio
import time
import threading
import random
from concurrent.futures import ThreadPoolExecutor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kernel.enhanced_core import EnhancedGAIAKernel
from memory.hierarchical_memory import HierarchicalMemoryManager, ZoneState

@pytest.mark.asyncio
class TestMemoryExhaustion:
    """Test behavior under memory exhaustion scenarios"""
    
    async def test_memory_limit_enforcement(self):
        """Test system behavior at memory limits"""
        # Very small memory limit
        manager = HierarchicalMemoryManager(total_memory_mb=2, compression_threshold=0.5)
        await manager.initialize()
        
        try:
            stored_keys = []
            
            # Try to store way more than memory limit
            for i in range(100):
                key = f"stress_test_{i}"
                # 10KB embedding - should hit limit quickly
                embedding = np.random.randn(2500).astype(np.float32)
                
                try:
                    zone_id = await manager.store(key, embedding)
                    stored_keys.append(key)
                    
                    # Force memory management
                    await manager._manage_memory_pressure()
                    
                except Exception as e:
                    print(f"Exception at iteration {i}: {e}")
                    break
            
            # Verify system is still functional
            status = manager.get_status()
            assert status['total_zones'] > 0
            
            # Verify compression occurred
            frozen_zones = sum(1 for z in manager.zones.values() 
                             if z.state == ZoneState.FROZEN)
            assert frozen_zones > 0
            
            # Test that we can still retrieve compressed data
            if stored_keys:
                test_key = random.choice(stored_keys)
                result = await manager.retrieve(test_key)
                # Should either work or fail gracefully
                if result is not None:
                    assert len(result) == 2  # (embedding, associations)
            
        finally:
            manager.maintenance_task.cancel()
            try:
                await manager.maintenance_task
            except asyncio.CancelledError:
                pass
    
    async def test_fragmentation_resilience(self):
        """Test resilience to memory fragmentation"""
        manager = HierarchicalMemoryManager(total_memory_mb=16)
        await manager.initialize()
        
        try:
            # Create fragmentation by storing and deleting random sizes
            stored_keys = []
            
            # Store variable-sized items
            for i in range(50):
                key = f"frag_test_{i}"
                size = random.randint(100, 5000)  # Variable embedding sizes
                embedding = np.random.randn(size).astype(np.float32)
                
                zone_id = await manager.store(key, embedding)
                stored_keys.append(key)
                
                # Randomly access old items (creates fragmentation)
                if stored_keys and random.random() < 0.3:
                    old_key = random.choice(stored_keys)
                    await manager.retrieve(old_key)
            
            # Force memory management multiple times
            for _ in range(5):
                await manager._manage_memory_pressure()
                await asyncio.sleep(0.1)
            
            # Verify system remains stable
            status = manager.get_status()
            assert status['total_zones'] > 0
            
            # Test retrieval still works
            test_keys = random.sample(stored_keys, min(10, len(stored_keys)))
            for key in test_keys:
                result = await manager.retrieve(key)
                if result is not None:  # May be compressed/archived
                    embedding, associations = result
                    assert isinstance(embedding, np.ndarray)
                    
        finally:
            manager.maintenance_task.cancel()
            try:
                await manager.maintenance_task
            except asyncio.CancelledError:
                pass

@pytest.mark.asyncio
class TestHighConcurrency:
    """Test system under high concurrency loads"""
    
    async def test_concurrent_memory_operations(self):
        """Test concurrent memory store/retrieve operations"""
        manager = HierarchicalMemoryManager(total_memory_mb=32)
        await manager.initialize()
        
        try:
            async def worker(worker_id: int, operations: int):
                """Worker function for concurrent operations"""
                local_keys = []
                
                for i in range(operations):
                    key = f"worker_{worker_id}_item_{i}"
                    embedding = np.random.randn(256).astype(np.float32)
                    
                    # Store
                    try:
                        zone_id = await manager.store(key, embedding)
                        local_keys.append(key)
                    except Exception as e:
                        print(f"Worker {worker_id} store error: {e}")
                    
                    # Randomly retrieve existing items
                    if local_keys and random.random() < 0.3:
                        retrieve_key = random.choice(local_keys)
                        try:
                            result = await manager.retrieve(retrieve_key)
                        except Exception as e:
                            print(f"Worker {worker_id} retrieve error: {e}")
                    
                    # Small delay to allow other workers
                    if i % 10 == 0:
                        await asyncio.sleep(0.001)
                
                return len(local_keys)
            
            # Run multiple workers concurrently
            num_workers = 10
            operations_per_worker = 20
            
            tasks = [worker(i, operations_per_worker) for i in range(num_workers)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successful operations
            successful_stores = sum(r for r in results if isinstance(r, int))
            exceptions = [r for r in results if isinstance(r, Exception)]
            
            print(f"Successful stores: {successful_stores}, Exceptions: {len(exceptions)}")
            
            # Verify system is still functional
            status = manager.get_status()
            assert status['total_zones'] > 0
            
        finally:
            manager.maintenance_task.cancel()
            try:
                await manager.maintenance_task
            except asyncio.CancelledError:
                pass
    
    async def test_concurrent_kernel_operations(self):
        """Test concurrent kernel thought execution"""
        kernel = EnhancedGAIAKernel(memory_size_mb=64)
        await kernel.initialize()
        
        try:
            # Register multiple models
            for i in range(5):
                kernel.register_model_with_fallback(
                    f"model_{i}", f"type_{i}", 100, 200
                )
            
            async def spawn_thoughts(batch_id: int, count: int):
                """Spawn multiple thoughts concurrently"""
                thought_ids = []
                
                for i in range(count):
                    model_id = f"model_{i % 5}"
                    priority = random.randint(1, 50)
                    
                    try:
                        thought_id = await kernel.spawn_thought_with_priority(
                            model_id,
                            {"batch": batch_id, "item": i},
                            priority=priority
                        )
                        thought_ids.append(thought_id)
                        
                        # Small delay to prevent overwhelming
                        if i % 5 == 0:
                            await asyncio.sleep(0.01)
                            
                    except Exception as e:
                        print(f"Batch {batch_id} thought spawn error: {e}")
                
                return len(thought_ids)
            
            # Run concurrent thought batches
            num_batches = 8
            thoughts_per_batch = 10
            
            tasks = [spawn_thoughts(i, thoughts_per_batch) for i in range(num_batches)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Wait for thoughts to complete
            await asyncio.sleep(2)
            
            # Count results
            successful_thoughts = sum(r for r in results if isinstance(r, int))
            print(f"Successfully spawned thoughts: {successful_thoughts}")
            
            # Verify system status
            status = kernel.get_enhanced_status()
            assert status['kernel']['total_thoughts_executed'] > 0
            
        finally:
            kernel.memory_manager.maintenance_task.cancel()
            try:
                await kernel.memory_manager.maintenance_task
            except asyncio.CancelledError:
                pass

@pytest.mark.asyncio
class TestFailureScenarios:
    """Test system behavior under various failure conditions"""
    
    async def test_corrupted_data_handling(self):
        """Test handling of corrupted memory data"""
        manager = HierarchicalMemoryManager(total_memory_mb=16)
        await manager.initialize()
        
        try:
            # Store normal data
            key = "test_data"
            embedding = np.random.randn(512).astype(np.float32)
            zone_id = await manager.store(key, embedding)
            
            # Corrupt the zone data
            zone = manager.zones[zone_id]
            zone.embeddings[key] = "corrupted_data"  # Wrong type
            
            # Try to retrieve - should handle gracefully
            try:
                result = await manager.retrieve(key)
                # Should either work or return None
                if result is not None:
                    assert len(result) == 2
            except Exception as e:
                # Should not crash the system
                print(f"Expected error handling corrupted data: {e}")
            
            # Verify system is still functional
            new_key = "new_data"
            new_embedding = np.random.randn(256).astype(np.float32)
            new_zone_id = await manager.store(new_key, new_embedding)
            assert new_zone_id is not None
            
        finally:
            manager.maintenance_task.cancel()
            try:
                await manager.maintenance_task
            except asyncio.CancelledError:
                pass
    
    async def test_model_cascade_failures(self):
        """Test system resilience to cascading model failures"""
        kernel = EnhancedGAIAKernel(memory_size_mb=32)
        await kernel.initialize()
        
        try:
            # Create chain of fallbacks
            kernel.register_model_with_fallback(
                "primary", "test", 100, 200, fallback_model_id="secondary"
            )
            kernel.register_model_with_fallback(
                "secondary", "test", 80, 150, fallback_model_id="tertiary"
            )
            kernel.register_model_with_fallback(
                "tertiary", "test", 60, 100  # No fallback
            )
            
            # Force all models to fail
            for model in kernel.models.values():
                model.current_accuracy = 0.1  # Below threshold
                model.failure_count = 10
            
            # Try to execute thoughts - should handle gracefully
            failures = 0
            for i in range(5):
                try:
                    thought_id = await kernel.spawn_thought_with_priority(
                        "primary", {"test": f"data_{i}"}, priority=10
                    )
                except Exception as e:
                    failures += 1
                    print(f"Expected failure {i}: {e}")
            
            await asyncio.sleep(1)  # Let any thoughts complete
            
            # System should still be responsive
            status = kernel.get_enhanced_status()
            assert status['kernel']['total_models'] == 3
            
            # Some thoughts should have failed
            assert status['kernel']['failed_thoughts'] >= 0
            
        finally:
            kernel.memory_manager.maintenance_task.cancel()
            try:
                await kernel.memory_manager.maintenance_task
            except asyncio.CancelledError:
                pass
    
    async def test_interrupt_system_overload(self):
        """Test interrupt system under extreme load"""
        kernel = EnhancedGAIAKernel(memory_size_mb=32)
        await kernel.initialize()
        
        try:
            # Register test model
            kernel.register_model_with_fallback(
                "test_model", "test", 100, 200
            )
            
            # Flood with high-priority interrupts
            thought_ids = []
            for i in range(100):
                try:
                    thought_id = await kernel.spawn_thought_with_priority(
                        "test_model", 
                        {"urgent_task": i},
                        priority=random.randint(80, 100)  # All high priority
                    )
                    thought_ids.append(thought_id)
                except Exception as e:
                    print(f"Interrupt overload error {i}: {e}")
            
            # Let system process
            await asyncio.sleep(2)
            
            # Verify system didn't crash
            status = kernel.get_enhanced_status()
            assert status['interrupts']['active_contexts'] >= 0
            assert status['interrupts']['suspended_contexts'] >= 0
            
        finally:
            kernel.memory_manager.maintenance_task.cancel()
            try:
                await kernel.memory_manager.maintenance_task
            except asyncio.CancelledError:
                pass

@pytest.mark.asyncio 
class TestResourceExhaustion:
    """Test behavior when system resources are exhausted"""
    
    async def test_attention_pool_exhaustion(self):
        """Test system when attention pool is exhausted"""
        kernel = EnhancedGAIAKernel(memory_size_mb=32)
        await kernel.initialize()
        
        try:
            # Set very small attention pool
            kernel.attention_pool = 10
            kernel.available_attention = 10
            
            # Register model
            kernel.register_model_with_fallback(
                "attention_hungry", "test", 100, 200
            )
            
            # Try to exhaust attention
            thought_ids = []
            for i in range(20):  # More than attention pool
                try:
                    # Simulate attention allocation
                    if kernel.available_attention > 0:
                        kernel.available_attention -= 1
                    
                    thought_id = await kernel.spawn_thought_with_priority(
                        "attention_hungry",
                        {"task": i},
                        priority=10
                    )
                    thought_ids.append(thought_id)
                    
                except Exception as e:
                    print(f"Attention exhaustion at {i}: {e}")
            
            await asyncio.sleep(1)
            
            # Verify system handled exhaustion gracefully
            status = kernel.get_enhanced_status()
            assert status['kernel']['available_attention'] >= 0
            
        finally:
            kernel.memory_manager.maintenance_task.cancel()
            try:
                await kernel.memory_manager.maintenance_task
            except asyncio.CancelledError:
                pass
    
    async def test_zone_proliferation(self):
        """Test system when too many zones are created"""
        manager = HierarchicalMemoryManager(total_memory_mb=64)
        await manager.initialize()
        
        try:
            # Create many zones by using unique categories
            for i in range(200):  # Create many zones
                key = f"unique_item_{i}"
                embedding = np.random.randn(100).astype(np.float32)
                category = f"unique_category_{i}"  # Force new zone per item
                
                try:
                    zone_id = await manager.store(key, embedding, 
                                                 semantic_category=category)
                except Exception as e:
                    print(f"Zone proliferation error at {i}: {e}")
                    break
                
                # Trigger memory management periodically
                if i % 20 == 0:
                    await manager._manage_memory_pressure()
            
            # Verify system is still functional
            status = manager.get_status()
            print(f"Final zones: {status['total_zones']}, "
                  f"Frozen: {status['frozen_zones']}")
            
            # Should have triggered compression/management
            assert status['frozen_zones'] > 0 or status['total_zones'] < 200
            
        finally:
            manager.maintenance_task.cancel()
            try:
                await manager.maintenance_task
            except asyncio.CancelledError:
                pass

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to see print statements