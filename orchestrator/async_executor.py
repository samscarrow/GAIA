"""
Async Thought Executor - Enables parallel, branching thought streams
"""

import asyncio
import uuid
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class ThoughtState(Enum):
    """States for thought execution"""
    PENDING = "pending"
    RUNNING = "running"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    YIELDED = "yielded"


@dataclass
class ThoughtNode:
    """Represents a unit of AI reasoning with async capabilities"""
    thought_id: str = field(default_factory=lambda: f"thought_{uuid.uuid4().hex[:8]}")
    model_id: str = ""
    parent_id: Optional[str] = None
    input_data: Any = None
    output_data: Any = None
    partial_outputs: List[Any] = field(default_factory=list)
    state: ThoughtState = ThoughtState.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    associations: Set[str] = field(default_factory=set)
    children: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def add_partial_output(self, output: Any):
        """Add partial output that other thoughts can read"""
        self.partial_outputs.append(output)
        
    def spawn_child(self, model_id: str, input_data: Any) -> 'ThoughtNode':
        """Create a child thought"""
        child = ThoughtNode(
            model_id=model_id,
            parent_id=self.thought_id,
            input_data=input_data,
            context=self.context.copy()
        )
        self.children.append(child.thought_id)
        return child


class AsyncThoughtExecutor:
    """
    Manages concurrent AI reasoning streams with branching and merging
    """
    
    def __init__(self, kernel=None):
        self.kernel = kernel
        self.thought_graph: Dict[str, ThoughtNode] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.suspended_thoughts: Dict[str, ThoughtNode] = {}
        self.output_streams: Dict[str, asyncio.Queue] = {}
        self.association_handlers: Dict[str, Callable] = {}
        self.interrupt_handlers: Dict[str, Callable] = {}
        
    async def spawn_thought(self, model_id: str, input_data: Any,
                          parent_id: Optional[str] = None,
                          context: Optional[Dict] = None) -> str:
        """Spawn a new thought stream asynchronously"""
        thought = ThoughtNode(
            model_id=model_id,
            parent_id=parent_id,
            input_data=input_data,
            context=context or {}
        )
        
        # Register in thought graph
        self.thought_graph[thought.thought_id] = thought
        
        # Create output stream for this thought
        self.output_streams[thought.thought_id] = asyncio.Queue()
        
        # Create and start async task
        task = asyncio.create_task(
            self._execute_thought(thought)
        )
        self.active_tasks[thought.thought_id] = task
        
        logger.info(f"Spawned thought {thought.thought_id} with model {model_id}")
        return thought.thought_id
        
    async def _execute_thought(self, thought: ThoughtNode):
        """Execute a thought with interruption points"""
        thought.state = ThoughtState.RUNNING
        thought.started_at = time.time()
        
        try:
            # Simulate model execution with yield points
            async for step_output in self._run_model_with_yields(thought):
                # Add partial output
                thought.add_partial_output(step_output)
                
                # Put output in stream for other thoughts
                await self.output_streams[thought.thought_id].put(step_output)
                
                # Check for associations to spawn
                associations = await self._detect_associations(thought, step_output)
                if associations:
                    for assoc_model, assoc_input in associations:
                        # Spawn associated thought without blocking
                        child_id = await self.spawn_thought(
                            assoc_model,
                            assoc_input,
                            parent_id=thought.thought_id,
                            context=thought.context
                        )
                        thought.associations.add(child_id)
                        logger.info(f"Thought {thought.thought_id} spawned association {child_id}")
                
                # Check for interrupt signals
                if await self._should_yield(thought):
                    thought.state = ThoughtState.YIELDED
                    await self._handle_yield(thought)
                    # Continue execution after yield
                    thought.state = ThoughtState.RUNNING
                    
            # Complete thought
            thought.state = ThoughtState.COMPLETED
            thought.completed_at = time.time()
            thought.output_data = thought.partial_outputs
            
            logger.info(f"Thought {thought.thought_id} completed")
            
        except Exception as e:
            thought.state = ThoughtState.FAILED
            logger.error(f"Thought {thought.thought_id} failed: {e}")
            raise
            
    async def _run_model_with_yields(self, thought: ThoughtNode):
        """Simulate model execution with yield points"""
        # This would be replaced with actual model execution
        for i in range(5):  # Simulate 5 steps
            await asyncio.sleep(0.1)  # Simulate processing
            
            output = {
                "step": i,
                "model": thought.model_id,
                "data": f"Output from step {i}",
                "timestamp": time.time()
            }
            
            yield output
            
    async def _detect_associations(self, thought: ThoughtNode, 
                                  output: Any) -> List[Tuple[str, Any]]:
        """Detect associations from partial output"""
        associations = []
        
        # Check if output triggers any associations
        if "trigger" in str(output).lower():
            # Example: spawn an analysis model
            associations.append(("analyzer", {"trigger_data": output}))
            
        return associations
        
    async def _should_yield(self, thought: ThoughtNode) -> bool:
        """Check if thought should yield to another"""
        # Check if there are higher priority thoughts waiting
        # This is a simplified check
        return False  # Placeholder
        
    async def _handle_yield(self, thought: ThoughtNode):
        """Handle yielding control"""
        # Save thought state
        self.suspended_thoughts[thought.thought_id] = thought
        
        # Release resources if using kernel
        if self.kernel:
            await self.kernel.release_attention(thought.model_id, 10)
            
        # Wait for signal to resume
        await asyncio.sleep(0.1)  # Placeholder
        
        # Restore state
        del self.suspended_thoughts[thought.thought_id]
        
    async def read_partial_output(self, thought_id: str, 
                                 timeout: Optional[float] = None) -> Any:
        """Read partial output from a running thought"""
        if thought_id not in self.output_streams:
            return None
            
        try:
            output = await asyncio.wait_for(
                self.output_streams[thought_id].get(),
                timeout=timeout
            )
            return output
        except asyncio.TimeoutError:
            return None
            
    async def wait_for_thought(self, thought_id: str) -> Any:
        """Wait for a thought to complete"""
        if thought_id not in self.active_tasks:
            return None
            
        task = self.active_tasks[thought_id]
        await task
        
        thought = self.thought_graph[thought_id]
        return thought.output_data
        
    async def merge_thought_streams(self, thought_ids: List[str]) -> Any:
        """Merge outputs from multiple thought streams"""
        results = []
        
        # Gather all results
        tasks = [self.wait_for_thought(tid) for tid in thought_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge logic
        merged = {
            "merged_from": thought_ids,
            "results": results,
            "timestamp": time.time()
        }
        
        return merged
        
    def get_thought_status(self, thought_id: str) -> Optional[Dict]:
        """Get status of a specific thought"""
        if thought_id not in self.thought_graph:
            return None
            
        thought = self.thought_graph[thought_id]
        return {
            "thought_id": thought.thought_id,
            "model_id": thought.model_id,
            "state": thought.state.value,
            "parent": thought.parent_id,
            "children": thought.children,
            "associations": list(thought.associations),
            "partial_outputs": len(thought.partial_outputs)
        }
        
    def visualize_thought_graph(self) -> Dict[str, Any]:
        """Get visualization data for thought graph"""
        nodes = []
        edges = []
        
        for thought_id, thought in self.thought_graph.items():
            nodes.append({
                "id": thought_id,
                "label": f"{thought.model_id}",
                "state": thought.state.value
            })
            
            # Parent edges
            if thought.parent_id:
                edges.append({
                    "from": thought.parent_id,
                    "to": thought_id,
                    "type": "parent"
                })
                
            # Association edges
            for assoc_id in thought.associations:
                edges.append({
                    "from": thought_id,
                    "to": assoc_id,
                    "type": "association"
                })
                
        return {"nodes": nodes, "edges": edges}