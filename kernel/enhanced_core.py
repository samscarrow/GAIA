"""
Enhanced GAIA Kernel with Hierarchical Memory Management
Integrates the production-ready memory system with the cognitive kernel
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import logging
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.hierarchical_memory_fixed import HierarchicalMemoryManager
from kernel.core import ModelState, CognitiveContext, ModelInstance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedModelInstance(ModelInstance):
    """Extended model instance with memory-aware features"""
    predicted_memory_usage: int = 0  # Predicted memory for next activation
    memory_zone_affinity: Set[str] = field(default_factory=set)  # Preferred memory zones
    fallback_model_id: Optional[str] = None  # Fallback for failures
    accuracy_threshold: float = 0.85  # Minimum acceptable accuracy
    current_accuracy: float = 1.0  # Current model accuracy
    failure_count: int = 0  # Number of failures
    
    def needs_replacement(self) -> bool:
        """Check if model needs replacement based on accuracy"""
        return self.current_accuracy < self.accuracy_threshold or self.failure_count > 3

class ModelLifecycleManager:
    """Manages model lifecycle with automatic fallback"""
    
    def __init__(self):
        self.model_versions: Dict[str, List[str]] = {}  # model_type -> [version_ids]
        self.accuracy_history: Dict[str, List[float]] = {}
        self.replacement_events = 0
        
    async def monitor_model(self, model: EnhancedModelInstance, result: Any) -> bool:
        """Monitor model performance and trigger replacement if needed"""
        # Simple accuracy simulation based on result
        if result is None or isinstance(result, Exception):
            model.failure_count += 1
            model.current_accuracy *= 0.9  # Degrade accuracy
        else:
            model.current_accuracy = min(1.0, model.current_accuracy * 1.02)  # Slowly recover
        
        # Track history
        if model.model_id not in self.accuracy_history:
            self.accuracy_history[model.model_id] = []
        self.accuracy_history[model.model_id].append(model.current_accuracy)
        
        # Check if replacement needed
        if model.needs_replacement():
            logger.warning(f"Model {model.model_id} needs replacement "
                         f"(accuracy: {model.current_accuracy:.2f}, failures: {model.failure_count})")
            return True
        
        return False
    
    async def replace_model(self, model: EnhancedModelInstance) -> Optional[EnhancedModelInstance]:
        """Replace failing model with fallback"""
        if not model.fallback_model_id:
            logger.error(f"No fallback for model {model.model_id}")
            return None
        
        logger.info(f"Replacing {model.model_id} with fallback {model.fallback_model_id}")
        
        # Create new instance with fallback
        new_model = EnhancedModelInstance(
            model_id=model.fallback_model_id,
            model_type=model.model_type,
            memory_footprint=model.memory_footprint,
            vram_required=model.vram_required,
            semantic_tags=model.semantic_tags,
            associations=model.associations,
            fallback_model_id=None,  # Prevent cascading fallbacks
            accuracy_threshold=model.accuracy_threshold
        )
        
        self.replacement_events += 1
        return new_model

@dataclass 
class InterruptContext:
    """Context for managing interruptible associations"""
    interrupt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 0  # Higher = more important
    preemptable: bool = True
    checkpoint_data: Optional[Dict[str, Any]] = None
    parent_thought_id: Optional[str] = None
    
    def can_preempt(self, other_priority: int) -> bool:
        """Check if this context can be preempted by another"""
        return self.preemptable and other_priority > self.priority

class PriorityInterruptManager:
    """Manages context-aware interrupts with priority-based preemption"""
    
    def __init__(self):
        self.active_contexts: Dict[str, InterruptContext] = {}
        self.suspended_contexts: List[InterruptContext] = []
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        
        # Priority levels
        self.PRIORITY_CRITICAL = 100  # System health, errors
        self.PRIORITY_USER = 50      # User interactions
        self.PRIORITY_BACKGROUND = 10  # Background processing
        
    async def request_interrupt(self, thought_id: str, priority: int,
                               checkpoint: Optional[Dict[str, Any]] = None) -> bool:
        """Request to interrupt current processing"""
        
        # Check if any active context can be preempted
        preemptable_contexts = [
            (tid, ctx) for tid, ctx in self.active_contexts.items()
            if ctx.can_preempt(priority)
        ]
        
        if not preemptable_contexts:
            logger.debug(f"Cannot interrupt: no preemptable contexts for priority {priority}")
            return False
        
        # Preempt lowest priority context
        target_id, target_ctx = min(preemptable_contexts, key=lambda x: x[1].priority)
        
        # Save checkpoint
        if checkpoint:
            self.checkpoints[target_id] = checkpoint
            target_ctx.checkpoint_data = checkpoint
        
        # Suspend context
        self.suspended_contexts.append(target_ctx)
        del self.active_contexts[target_id]
        
        logger.info(f"Preempted thought {target_id} (priority {target_ctx.priority}) "
                   f"for {thought_id} (priority {priority})")
        
        # Create new context
        new_context = InterruptContext(
            priority=priority,
            parent_thought_id=target_id,
            preemptable=(priority < self.PRIORITY_CRITICAL)
        )
        self.active_contexts[thought_id] = new_context
        
        return True
    
    async def resume_context(self, thought_id: str) -> Optional[Dict[str, Any]]:
        """Resume a suspended context"""
        # Find suspended context
        for i, ctx in enumerate(self.suspended_contexts):
            if ctx.parent_thought_id == thought_id:
                # Restore context
                self.active_contexts[thought_id] = ctx
                self.suspended_contexts.pop(i)
                
                # Return checkpoint
                return self.checkpoints.get(thought_id)
        
        return None

class EnhancedGAIAKernel:
    """Production-ready GAIA kernel with all critical refinements"""
    
    def __init__(self, memory_size_mb: int = 1024):
        # Core components
        self.models: Dict[str, EnhancedModelInstance] = {}
        self.contexts: Dict[str, CognitiveContext] = {}
        self.active_context: Optional[CognitiveContext] = None
        
        # Enhanced components
        self.memory_manager = HierarchicalMemoryManager(
            total_memory_mb=memory_size_mb,
            compression_threshold=0.7
        )
        self.lifecycle_manager = ModelLifecycleManager()
        self.interrupt_manager = PriorityInterruptManager()
        
        # Resource tracking
        self.attention_pool: int = 100
        self.available_attention: int = 100
        self.thought_streams: Dict[str, asyncio.Task] = {}
        
        # Performance metrics
        self.total_thoughts_executed = 0
        self.failed_thoughts = 0
        self.preempted_thoughts = 0
        
    async def initialize(self):
        """Initialize the enhanced kernel"""
        logger.info("Initializing Enhanced GAIA Kernel...")
        
        # Initialize memory manager if method exists
        if hasattr(self.memory_manager, 'initialize'):
            await self.memory_manager.initialize()
        
        # Create initial context
        self.active_context = CognitiveContext()
        self.contexts[self.active_context.context_id] = self.active_context
        
        # Start monitoring
        asyncio.create_task(self._monitor_system_health())
        
        logger.info(f"Enhanced GAIA Kernel initialized with {self.memory_manager.total_memory_bytes / (1024*1024):.1f}MB memory")
    
    def register_model_with_fallback(self, model_id: str, model_type: str,
                                    memory_footprint: int, vram_required: int,
                                    fallback_model_id: Optional[str] = None,
                                    semantic_tags: Optional[Set[str]] = None):
        """Register model with fallback support"""
        model = EnhancedModelInstance(
            model_id=model_id,
            model_type=model_type,
            memory_footprint=memory_footprint,
            vram_required=vram_required,
            semantic_tags=semantic_tags or set(),
            fallback_model_id=fallback_model_id,
            predicted_memory_usage=memory_footprint
        )
        self.models[model_id] = model
        logger.info(f"Registered model {model_id} with fallback {fallback_model_id}")
    
    async def spawn_thought_with_priority(self, model_id: str, input_data: Any,
                                         priority: int = 10,
                                         context: Optional[CognitiveContext] = None) -> str:
        """Spawn thought with priority-based execution"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not registered")
        
        thought_id = f"thought_{uuid.uuid4().hex[:8]}"
        context = context or self.active_context
        
        # Request interrupt if high priority
        if priority > self.interrupt_manager.PRIORITY_BACKGROUND:
            checkpoint = await self._create_checkpoint()
            await self.interrupt_manager.request_interrupt(thought_id, priority, checkpoint)
        
        # Create task
        task = asyncio.create_task(
            self._execute_enhanced_thought(thought_id, model_id, input_data, context, priority)
        )
        self.thought_streams[thought_id] = task
        
        logger.info(f"Spawned priority {priority} thought {thought_id} with model {model_id}")
        return thought_id
    
    async def _execute_enhanced_thought(self, thought_id: str, model_id: str,
                                       input_data: Any, context: CognitiveContext,
                                       priority: int):
        """Execute thought with enhanced features"""
        model = self.models[model_id]
        model.state = ModelState.ACTIVE
        model.last_accessed = time.time()
        
        try:
            # Store thought context in memory
            thought_embedding = np.random.randn(512).astype(np.float32)  # Simulate embedding
            await self.memory_manager.store(
                key=thought_id,
                embedding=thought_embedding,
                associations={model_id},
                semantic_category=model.model_type
            )
            
            # Simulate model execution
            await asyncio.sleep(0.1)
            result = {"thought_id": thought_id, "result": f"Processed by {model_id}"}
            
            # Monitor model performance
            needs_replacement = await self.lifecycle_manager.monitor_model(model, result)
            
            if needs_replacement:
                # Replace with fallback
                new_model = await self.lifecycle_manager.replace_model(model)
                if new_model:
                    self.models[new_model.model_id] = new_model
                    # Retry with new model
                    return await self._execute_enhanced_thought(
                        thought_id, new_model.model_id, input_data, context, priority
                    )
            
            # Check for associations
            associations = model.associations
            for associated_model in associations:
                if associated_model in context.active_associations:
                    # Spawn associated thought
                    child_context = context.spawn_child()
                    await self.spawn_thought_with_priority(
                        associated_model,
                        {"parent": input_data, "association": True},
                        priority=priority - 5,  # Lower priority for associations
                        context=child_context
                    )
            
            model.state = ModelState.DORMANT
            self.total_thoughts_executed += 1
            
            # Store result in memory
            result_embedding = np.random.randn(512).astype(np.float32)
            await self.memory_manager.store(
                key=f"{thought_id}_result",
                embedding=result_embedding,
                associations={thought_id, model_id}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in thought {thought_id}: {e}")
            model.state = ModelState.DORMANT
            model.failure_count += 1
            self.failed_thoughts += 1
            raise
    
    async def _create_checkpoint(self) -> Dict[str, Any]:
        """Create checkpoint of current state"""
        return {
            'timestamp': time.time(),
            'active_thoughts': list(self.thought_streams.keys()),
            'memory_status': self.memory_manager.get_status(),
            'attention_available': self.available_attention
        }
    
    async def _monitor_system_health(self):
        """Monitor system health and trigger high-priority actions"""
        while True:
            await asyncio.sleep(10)
            
            try:
                # Check memory pressure
                status = self.memory_manager.get_status()
                memory_usage = status['memory_used_mb'] / (self.memory_manager.total_memory_bytes / (1024*1024))
                
                if memory_usage > 0.9:
                    # Critical memory pressure
                    logger.warning(f"Critical memory pressure: {memory_usage:.1%}")
                    await self.spawn_thought_with_priority(
                        "system_memory_manager",
                        {"action": "emergency_cleanup"},
                        priority=self.interrupt_manager.PRIORITY_CRITICAL
                    )
                
                # Check model health
                for model in self.models.values():
                    if model.needs_replacement():
                        logger.warning(f"Model {model.model_id} health check failed")
                
                # Log metrics
                logger.info(f"System Health: {self.total_thoughts_executed} thoughts executed, "
                          f"{self.failed_thoughts} failed, {self.preempted_thoughts} preempted, "
                          f"Memory: {status['memory_used_mb']:.1f}MB, "
                          f"Cache hit rate: {status['cache_hit_rate']:.1%}")
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        memory_status = self.memory_manager.get_status()
        
        return {
            "kernel": {
                "active_models": sum(1 for m in self.models.values() if m.state == ModelState.ACTIVE),
                "total_models": len(self.models),
                "failing_models": sum(1 for m in self.models.values() if m.needs_replacement()),
                "active_thoughts": len([t for t in self.thought_streams.values() if not t.done()]),
                "total_thoughts_executed": self.total_thoughts_executed,
                "failed_thoughts": self.failed_thoughts,
                "preempted_thoughts": self.preempted_thoughts,
                "available_attention": self.available_attention,
            },
            "memory": memory_status,
            "lifecycle": {
                "model_replacements": self.lifecycle_manager.replacement_events,
                "accuracy_tracking": len(self.lifecycle_manager.accuracy_history)
            },
            "interrupts": {
                "active_contexts": len(self.interrupt_manager.active_contexts),
                "suspended_contexts": len(self.interrupt_manager.suspended_contexts)
            }
        }

async def demonstrate_enhanced_kernel():
    """Demonstration of the enhanced kernel capabilities"""
    logger.info("=" * 60)
    logger.info("ENHANCED GAIA KERNEL DEMONSTRATION")
    logger.info("=" * 60)
    
    # Initialize kernel
    kernel = EnhancedGAIAKernel(memory_size_mb=512)
    await kernel.initialize()
    
    # Register models with fallbacks
    kernel.register_model_with_fallback(
        "vision_primary", "vision", 200, 1000,
        fallback_model_id="vision_backup",
        semantic_tags={"perception", "visual"}
    )
    kernel.register_model_with_fallback(
        "vision_backup", "vision", 150, 800,
        semantic_tags={"perception", "visual", "lightweight"}
    )
    kernel.register_model_with_fallback(
        "reasoning_primary", "reasoning", 300, 1500,
        fallback_model_id="reasoning_backup",
        semantic_tags={"logic", "inference"}
    )
    kernel.register_model_with_fallback(
        "system_memory_manager", "system", 50, 100,
        semantic_tags={"system", "maintenance"}
    )
    
    # Create associations
    kernel.models["vision_primary"].associations.add("reasoning_primary")
    
    # Simulate workload
    logger.info("\nSimulating mixed-priority workload...")
    
    thoughts = []
    
    # Low priority background task
    thoughts.append(await kernel.spawn_thought_with_priority(
        "vision_primary",
        {"task": "background_processing"},
        priority=5
    ))
    
    # User interaction (higher priority)
    await asyncio.sleep(0.5)
    thoughts.append(await kernel.spawn_thought_with_priority(
        "reasoning_primary",
        {"task": "user_query", "query": "What is the meaning of life?"},
        priority=kernel.interrupt_manager.PRIORITY_USER
    ))
    
    # Critical system task
    await asyncio.sleep(0.5)
    thoughts.append(await kernel.spawn_thought_with_priority(
        "system_memory_manager",
        {"task": "emergency_cleanup"},
        priority=kernel.interrupt_manager.PRIORITY_CRITICAL
    ))
    
    # Wait for completion
    await asyncio.sleep(2)
    
    # Get final status
    status = kernel.get_enhanced_status()
    
    logger.info("\n" + "=" * 60)
    logger.info("DEMONSTRATION RESULTS")
    logger.info("=" * 60)
    
    logger.info("\nKernel Status:")
    for key, value in status['kernel'].items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\nMemory Status:")
    for key, value in status['memory'].items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")
    
    logger.info("\nLifecycle Status:")
    for key, value in status['lifecycle'].items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\nInterrupt Status:")
    for key, value in status['interrupts'].items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\n✅ Enhanced GAIA Kernel demonstration complete!")

if __name__ == "__main__":
    asyncio.run(demonstrate_enhanced_kernel())