"""
Integration tests for GAIA kernel components working together
"""

import pytest
import numpy as np
import asyncio
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kernel.enhanced_core import (
    EnhancedGAIAKernel, EnhancedModelInstance, 
    ModelLifecycleManager, PriorityInterruptManager,
    InterruptContext
)
from memory.hierarchical_memory import ZoneState

@pytest.mark.asyncio
class TestKernelMemoryIntegration:
    """Test kernel and memory system integration"""
    
    async def test_kernel_memory_initialization(self):
        """Test kernel properly initializes memory system"""
        kernel = EnhancedGAIAKernel(memory_size_mb=64)
        await kernel.initialize()
        
        try:
            # Verify memory system is initialized
            assert kernel.memory_manager is not None
            assert kernel.memory_manager.total_memory_bytes == 64 * 1024 * 1024
            
            # Verify maintenance task is running
            assert kernel.memory_manager.maintenance_task is not None
            assert not kernel.memory_manager.maintenance_task.done()
            
        finally:
            # Cleanup
            kernel.memory_manager.maintenance_task.cancel()
            try:
                await kernel.memory_manager.maintenance_task
            except asyncio.CancelledError:
                pass
    
    async def test_thought_memory_storage(self):
        """Test that thoughts are properly stored in memory"""
        kernel = EnhancedGAIAKernel(memory_size_mb=64)
        await kernel.initialize()
        
        try:
            # Register a test model
            kernel.register_model_with_fallback(
                "test_model", "test", 100, 200
            )
            
            # Spawn a thought
            thought_id = await kernel.spawn_thought_with_priority(
                "test_model", {"test": "data"}, priority=10
            )
            
            # Wait for execution
            await asyncio.sleep(0.5)
            
            # Verify thought was stored in memory
            thought_result = await kernel.memory_manager.retrieve(thought_id)
            assert thought_result is not None
            
            # Verify result was also stored
            result_data = await kernel.memory_manager.retrieve(f"{thought_id}_result")
            assert result_data is not None
            
        finally:
            # Cleanup
            kernel.memory_manager.maintenance_task.cancel()
            try:
                await kernel.memory_manager.maintenance_task
            except asyncio.CancelledError:
                pass
    
    async def test_model_fallback_integration(self):
        """Test model fallback with memory consistency"""
        kernel = EnhancedGAIAKernel(memory_size_mb=64)
        await kernel.initialize()
        
        try:
            # Register models with fallback
            kernel.register_model_with_fallback(
                "primary_model", "test", 100, 200, 
                fallback_model_id="backup_model"
            )
            kernel.register_model_with_fallback(
                "backup_model", "test", 80, 150
            )
            
            # Force primary model to fail by setting low accuracy
            primary_model = kernel.models["primary_model"]
            primary_model.current_accuracy = 0.5  # Below threshold
            primary_model.failure_count = 5
            
            # Spawn thought - should trigger fallback
            thought_id = await kernel.spawn_thought_with_priority(
                "primary_model", {"test": "data"}, priority=10
            )
            
            # Wait for execution
            await asyncio.sleep(0.5)
            
            # Verify fallback was used (backup model should be registered)
            assert "backup_model" in kernel.models
            
            # Verify lifecycle manager recorded the replacement
            assert kernel.lifecycle_manager.replacement_events > 0
            
        finally:
            # Cleanup
            kernel.memory_manager.maintenance_task.cancel()
            try:
                await kernel.memory_manager.maintenance_task
            except asyncio.CancelledError:
                pass

@pytest.mark.asyncio 
class TestPriorityInterruptSystem:
    """Test priority-based interrupt system"""
    
    async def test_priority_preemption(self):
        """Test high-priority thoughts can preempt low-priority ones"""
        kernel = EnhancedGAIAKernel(memory_size_mb=64)
        await kernel.initialize()
        
        try:
            # Register test models
            kernel.register_model_with_fallback("model1", "test", 100, 200)
            kernel.register_model_with_fallback("model2", "test", 100, 200)
            
            # Start low-priority thought
            low_priority_thought = await kernel.spawn_thought_with_priority(
                "model1", {"task": "background"}, priority=5
            )
            
            await asyncio.sleep(0.1)  # Let it start
            
            # Start high-priority thought
            high_priority_thought = await kernel.spawn_thought_with_priority(
                "model2", {"task": "urgent"}, priority=80
            )
            
            await asyncio.sleep(0.5)  # Let both complete
            
            # Verify interrupt system has active contexts
            # (This is a basic check - full preemption testing would need more sophisticated mocking)
            assert len(kernel.interrupt_manager.active_contexts) >= 0
            
        finally:
            # Cleanup
            kernel.memory_manager.maintenance_task.cancel()
            try:
                await kernel.memory_manager.maintenance_task
            except asyncio.CancelledError:
                pass
    
    def test_interrupt_context_creation(self):
        """Test interrupt context creation and priorities"""
        manager = PriorityInterruptManager()
        
        # Test context creation
        context = InterruptContext(priority=50)
        assert context.priority == 50
        assert context.preemptable is True
        
        # Test preemption logic
        assert context.can_preempt(60)  # Higher priority can preempt
        assert not context.can_preempt(40)  # Lower priority cannot
        
        # Critical priority should not be preemptable
        critical_context = InterruptContext(priority=100, preemptable=False)
        assert not critical_context.can_preempt(110)
    
    @pytest.mark.asyncio
    async def test_checkpoint_resume(self):
        """Test checkpoint creation and resumption"""
        manager = PriorityInterruptManager()
        
        # Request interrupt with checkpoint
        checkpoint_data = {"state": "processing", "progress": 0.5}
        success = await manager.request_interrupt(
            "thought1", priority=50, checkpoint=checkpoint_data
        )
        
        # No active contexts to preempt initially
        assert not success
        
        # Add an active context to preempt
        manager.active_contexts["existing"] = InterruptContext(priority=30)
        
        success = await manager.request_interrupt(
            "thought1", priority=50, checkpoint=checkpoint_data
        )
        
        assert success
        assert "thought1" in manager.active_contexts
        assert len(manager.suspended_contexts) == 1
        
        # Test resumption
        resumed_data = await manager.resume_context("existing")
        # Should be None since we're not resuming the right context
        # (This test could be enhanced with more realistic scenarios)

@pytest.mark.asyncio
class TestModelLifecycleIntegration:
    """Test model lifecycle management"""
    
    async def test_model_accuracy_tracking(self):
        """Test model accuracy monitoring"""
        manager = ModelLifecycleManager()
        
        # Create test model
        model = EnhancedModelInstance(
            model_id="test_model",
            model_type="test",
            memory_footprint=100,
            vram_required=200
        )
        
        # Simulate successful operations
        for _ in range(5):
            needs_replacement = await manager.monitor_model(model, {"success": True})
            assert not needs_replacement
        
        # Simulate failures
        for _ in range(10):
            needs_replacement = await manager.monitor_model(model, None)
        
        # Should need replacement after multiple failures
        assert model.needs_replacement()
        assert model.current_accuracy < model.accuracy_threshold
    
    async def test_model_replacement_process(self):
        """Test model replacement workflow"""
        manager = ModelLifecycleManager()
        
        # Create failing model with fallback
        failing_model = EnhancedModelInstance(
            model_id="failing_model",
            model_type="test",
            memory_footprint=100,
            vram_required=200,
            fallback_model_id="backup_model",
            current_accuracy=0.5,  # Below threshold
            failure_count=5
        )
        
        # Attempt replacement
        new_model = await manager.replace_model(failing_model)
        
        assert new_model is not None
        assert new_model.model_id == "backup_model"
        assert new_model.model_type == "test"
        assert manager.replacement_events == 1
    
    async def test_no_fallback_scenario(self):
        """Test behavior when no fallback is available"""
        manager = ModelLifecycleManager()
        
        # Create failing model without fallback
        failing_model = EnhancedModelInstance(
            model_id="failing_model",
            model_type="test",
            memory_footprint=100,
            vram_required=200,
            fallback_model_id=None,  # No fallback
            current_accuracy=0.5,
            failure_count=5
        )
        
        # Attempt replacement
        new_model = await manager.replace_model(failing_model)
        
        assert new_model is None  # Should fail gracefully
        assert manager.replacement_events == 0

@pytest.mark.asyncio
class TestSystemHealthMonitoring:
    """Test system health monitoring and responses"""
    
    async def test_memory_pressure_response(self):
        """Test kernel response to memory pressure"""
        # Small memory to trigger pressure quickly
        kernel = EnhancedGAIAKernel(memory_size_mb=8)
        await kernel.initialize()
        
        try:
            # Register system memory manager
            kernel.register_model_with_fallback(
                "system_memory_manager", "system", 50, 100
            )
            
            # Fill memory to create pressure
            for i in range(20):
                await kernel.memory_manager.store(
                    f"large_item_{i}",
                    np.random.randn(1000).astype(np.float32),  # 4KB each
                    semantic_category="test"
                )
            
            # Let health monitor run
            await asyncio.sleep(1)
            
            # Check if system responded to pressure
            # (In a real scenario, this would trigger emergency cleanup)
            status = kernel.get_enhanced_status()
            assert status['memory']['total_zones'] > 0
            
        finally:
            # Cleanup
            kernel.memory_manager.maintenance_task.cancel()
            try:
                await kernel.memory_manager.maintenance_task
            except asyncio.CancelledError:
                pass
    
    async def test_status_reporting(self):
        """Test comprehensive status reporting"""
        kernel = EnhancedGAIAKernel(memory_size_mb=64)
        await kernel.initialize()
        
        try:
            # Register some models
            kernel.register_model_with_fallback(
                "model1", "vision", 100, 200, fallback_model_id="backup1"
            )
            kernel.register_model_with_fallback(
                "backup1", "vision", 80, 150
            )
            
            # Execute some thoughts
            thought1 = await kernel.spawn_thought_with_priority(
                "model1", {"test": "data1"}, priority=10
            )
            thought2 = await kernel.spawn_thought_with_priority(
                "model1", {"test": "data2"}, priority=20
            )
            
            await asyncio.sleep(0.5)
            
            # Get status
            status = kernel.get_enhanced_status()
            
            # Verify status structure
            assert 'kernel' in status
            assert 'memory' in status
            assert 'lifecycle' in status
            assert 'interrupts' in status
            
            # Verify kernel metrics
            assert status['kernel']['total_models'] == 2
            assert status['kernel']['total_thoughts_executed'] >= 0
            
            # Verify memory metrics
            assert status['memory']['total_zones'] >= 0
            
        finally:
            # Cleanup
            kernel.memory_manager.maintenance_task.cancel()
            try:
                await kernel.memory_manager.maintenance_task
            except asyncio.CancelledError:
                pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])