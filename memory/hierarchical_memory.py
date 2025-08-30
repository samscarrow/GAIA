"""
Hierarchical Memory Management System for GAIA
Implements the #1 refinement: semantic zone partitioning with automatic management
"""

import asyncio
import time
import zlib
import pickle
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import OrderedDict
import heapq
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZoneState(Enum):
    """States for memory zones"""
    ACTIVE = "active"      # Currently in use
    WARM = "warm"          # Recently accessed
    COLD = "cold"          # Not recently accessed
    FROZEN = "frozen"      # Compressed, needs thawing
    ARCHIVED = "archived"  # Moved to long-term storage

@dataclass
class MemoryZone:
    """Represents a semantic memory partition (100-500KB target)"""
    zone_id: str
    semantic_category: str
    state: ZoneState = ZoneState.WARM
    size_bytes: int = 0
    max_size_bytes: int = 512 * 1024  # 512KB max
    min_size_bytes: int = 100 * 1024  # 100KB min
    
    # Content storage
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    associations: Dict[str, Set[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    compressed_data: Optional[bytes] = None
    
    # Access tracking
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    access_frequency: float = 0.0
    creation_time: float = field(default_factory=time.time)
    
    # Performance metrics
    avg_access_time_ms: float = 0.0
    compression_ratio: float = 1.0
    
    def calculate_priority(self) -> float:
        """Calculate zone priority for resource allocation"""
        recency = 1.0 / (time.time() - self.last_accessed + 1)
        frequency = self.access_frequency
        size_factor = 1.0 - (self.size_bytes / self.max_size_bytes)
        return (recency * 0.4) + (frequency * 0.4) + (size_factor * 0.2)
    
    def should_freeze(self, idle_threshold_seconds: float = 300) -> bool:
        """Determine if zone should be frozen"""
        idle_time = time.time() - self.last_accessed
        return (idle_time > idle_threshold_seconds and 
                self.state in [ZoneState.COLD, ZoneState.WARM] and
                self.size_bytes > self.min_size_bytes)
    
    def should_split(self) -> bool:
        """Check if zone should be split into smaller zones"""
        return self.size_bytes > self.max_size_bytes * 0.9
    
    def should_merge(self) -> bool:
        """Check if zone is too small and should be merged"""
        return self.size_bytes < self.min_size_bytes * 0.5

class BTreeNode:
    """B-Tree node for efficient zone indexing"""
    def __init__(self, leaf: bool = False, t: int = 3):
        self.keys: List[str] = []
        self.values: List[str] = []  # zone_ids
        self.children: List['BTreeNode'] = []
        self.leaf = leaf
        self.t = t  # Minimum degree
        
    def is_full(self) -> bool:
        return len(self.keys) == 2 * self.t - 1

class SemanticBTree:
    """B-Tree for semantic key to zone mapping"""
    def __init__(self, t: int = 3):
        self.root = BTreeNode(leaf=True, t=t)
        self.t = t
        
    def search(self, key: str) -> Optional[str]:
        """Search for zone_id by semantic key"""
        return self._search_node(self.root, key)
    
    def _search_node(self, node: BTreeNode, key: str) -> Optional[str]:
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
            
        if i < len(node.keys) and key == node.keys[i]:
            return node.values[i]
        
        if node.leaf:
            return None
            
        return self._search_node(node.children[i], key)
    
    def insert(self, key: str, zone_id: str):
        """Insert semantic key to zone mapping"""
        if self.root.is_full():
            new_root = BTreeNode(t=self.t)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root
        self._insert_non_full(self.root, key, zone_id)
    
    def _insert_non_full(self, node: BTreeNode, key: str, zone_id: str):
        i = len(node.keys) - 1
        
        if node.leaf:
            node.keys.append(None)
            node.values.append(None)
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                node.values[i + 1] = node.values[i]
                i -= 1
            node.keys[i + 1] = key
            node.values[i + 1] = zone_id
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            
            if node.children[i].is_full():
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            
            self._insert_non_full(node.children[i], key, zone_id)
    
    def _split_child(self, parent: BTreeNode, index: int):
        t = self.t
        full_child = parent.children[index]
        new_child = BTreeNode(leaf=full_child.leaf, t=t)
        
        mid_index = t - 1
        
        # Ensure we have enough keys to split
        if len(full_child.keys) <= mid_index:
            return
            
        # Save the middle key before modifying
        mid_key = full_child.keys[mid_index]
        mid_value = full_child.values[mid_index]
        
        new_child.keys = full_child.keys[mid_index + 1:]
        new_child.values = full_child.values[mid_index + 1:]
        full_child.keys = full_child.keys[:mid_index]
        full_child.values = full_child.values[:mid_index]
        
        if not full_child.leaf:
            new_child.children = full_child.children[mid_index + 1:]
            full_child.children = full_child.children[:mid_index + 1]
        
        parent.keys.insert(index, mid_key)
        parent.values.insert(index, mid_value)
        parent.children.insert(index + 1, new_child)

class HierarchicalMemoryManager:
    """Main memory manager implementing hierarchical partitioning"""
    
    def __init__(self, total_memory_mb: int = 1024, compression_threshold: float = 0.7):
        self.total_memory_bytes = total_memory_mb * 1024 * 1024
        self.compression_threshold = compression_threshold
        self.zones: Dict[str, MemoryZone] = {}
        self.semantic_index = SemanticBTree()
        self.zone_hierarchy: Dict[str, List[str]] = {}  # parent -> children
        
        # Memory tracking
        self.used_memory_bytes = 0
        self.compressed_memory_bytes = 0
        self.active_zones: Set[str] = set()
        
        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.compression_events = 0
        self.decompression_events = 0
        
        # Background tasks
        self.maintenance_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the memory manager"""
        logger.info(f"Initializing Hierarchical Memory Manager with {self.total_memory_bytes / (1024*1024):.1f}MB")
        self.maintenance_task = asyncio.create_task(self._maintenance_loop())
        
    async def store(self, key: str, embedding: np.ndarray, 
                   associations: Optional[Set[str]] = None,
                   semantic_category: Optional[str] = None) -> str:
        """Store data in appropriate zone"""
        semantic_category = semantic_category or self._infer_category(embedding)
        zone = await self._get_or_create_zone(semantic_category)
        
        # Track access
        zone.last_accessed = time.time()
        zone.access_count += 1
        
        # Store in zone
        zone.embeddings[key] = embedding
        if associations:
            zone.associations[key] = associations
        
        # Update size
        zone.size_bytes += embedding.nbytes
        
        # Update index
        self.semantic_index.insert(key, zone.zone_id)
        
        # Check if zone needs splitting
        if zone.should_split():
            await self._split_zone(zone)
        
        # Check memory pressure
        await self._manage_memory_pressure()
        
        return zone.zone_id
    
    async def retrieve(self, key: str) -> Optional[Tuple[np.ndarray, Set[str]]]:
        """Retrieve data from zones"""
        zone_id = self.semantic_index.search(key)
        if not zone_id:
            self.cache_misses += 1
            return None
        
        zone = self.zones.get(zone_id)
        if not zone:
            self.cache_misses += 1
            return None
        
        # Thaw if frozen
        if zone.state == ZoneState.FROZEN:
            await self._thaw_zone(zone)
        
        # Update access tracking
        zone.last_accessed = time.time()
        zone.access_count += 1
        zone.state = ZoneState.ACTIVE
        self.active_zones.add(zone_id)
        
        self.cache_hits += 1
        
        embedding = zone.embeddings.get(key)
        associations = zone.associations.get(key, set())
        
        return (embedding, associations) if embedding is not None else None
    
    async def _get_or_create_zone(self, semantic_category: str) -> MemoryZone:
        """Get existing zone or create new one"""
        # Look for existing zone with capacity
        for zone in self.zones.values():
            if (zone.semantic_category == semantic_category and 
                zone.size_bytes < zone.max_size_bytes * 0.8 and
                zone.state != ZoneState.FROZEN):
                return zone
        
        # Create new zone
        zone_id = f"zone_{semantic_category}_{len(self.zones)}"
        zone = MemoryZone(
            zone_id=zone_id,
            semantic_category=semantic_category
        )
        self.zones[zone_id] = zone
        logger.info(f"Created new zone: {zone_id}")
        return zone
    
    def _infer_category(self, embedding: np.ndarray) -> str:
        """Infer semantic category from embedding"""
        # Simple clustering based on embedding magnitude ranges
        magnitude = np.linalg.norm(embedding)
        if magnitude < 0.3:
            return "low_activation"
        elif magnitude < 0.7:
            return "medium_activation"
        else:
            return "high_activation"
    
    async def _freeze_zone(self, zone: MemoryZone):
        """Compress and freeze a zone"""
        if zone.state == ZoneState.FROZEN:
            return
        
        start_time = time.time()
        
        # Serialize zone data
        data = {
            'embeddings': zone.embeddings,
            'associations': zone.associations,
            'metadata': zone.metadata
        }
        serialized = pickle.dumps(data)
        
        # Compress
        zone.compressed_data = zlib.compress(serialized, level=6)
        zone.compression_ratio = len(zone.compressed_data) / len(serialized)
        
        # Clear original data
        zone.embeddings.clear()
        zone.associations.clear()
        zone.metadata.clear()
        
        # Update state
        zone.state = ZoneState.FROZEN
        self.compressed_memory_bytes += len(zone.compressed_data)
        self.compression_events += 1
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Froze zone {zone.zone_id} in {elapsed_ms:.1f}ms, "
                   f"compression ratio: {zone.compression_ratio:.2f}")
    
    async def _thaw_zone(self, zone: MemoryZone):
        """Decompress and thaw a zone"""
        if zone.state != ZoneState.FROZEN or not zone.compressed_data:
            return
        
        start_time = time.time()
        
        # Decompress
        serialized = zlib.decompress(zone.compressed_data)
        data = pickle.loads(serialized)
        
        # Restore data
        zone.embeddings = data['embeddings']
        zone.associations = data['associations']
        zone.metadata = data['metadata']
        
        # Clear compressed data
        self.compressed_memory_bytes -= len(zone.compressed_data)
        zone.compressed_data = None
        
        # Update state
        zone.state = ZoneState.WARM
        self.decompression_events += 1
        
        elapsed_ms = (time.time() - start_time) * 1000
        zone.avg_access_time_ms = (zone.avg_access_time_ms + elapsed_ms) / 2
        logger.info(f"Thawed zone {zone.zone_id} in {elapsed_ms:.1f}ms")
    
    async def _split_zone(self, zone: MemoryZone):
        """Split oversized zone into smaller zones"""
        logger.info(f"Splitting zone {zone.zone_id} (size: {zone.size_bytes / 1024:.1f}KB)")
        
        # Create two child zones
        child1 = MemoryZone(
            zone_id=f"{zone.zone_id}_1",
            semantic_category=zone.semantic_category
        )
        child2 = MemoryZone(
            zone_id=f"{zone.zone_id}_2",
            semantic_category=zone.semantic_category
        )
        
        # Distribute content
        items = list(zone.embeddings.items())
        mid = len(items) // 2
        
        for key, embedding in items[:mid]:
            child1.embeddings[key] = embedding
            child1.associations[key] = zone.associations.get(key, set())
            child1.size_bytes += embedding.nbytes
            self.semantic_index.insert(key, child1.zone_id)
        
        for key, embedding in items[mid:]:
            child2.embeddings[key] = embedding
            child2.associations[key] = zone.associations.get(key, set())
            child2.size_bytes += embedding.nbytes
            self.semantic_index.insert(key, child2.zone_id)
        
        # Register new zones
        self.zones[child1.zone_id] = child1
        self.zones[child2.zone_id] = child2
        
        # Update hierarchy
        self.zone_hierarchy[zone.zone_id] = [child1.zone_id, child2.zone_id]
        
        # Remove old zone
        del self.zones[zone.zone_id]
    
    async def _manage_memory_pressure(self):
        """Handle memory pressure by freezing/archiving zones"""
        memory_usage_ratio = self.used_memory_bytes / self.total_memory_bytes
        
        if memory_usage_ratio > self.compression_threshold:
            logger.warning(f"Memory pressure detected: {memory_usage_ratio:.1%} used")
            
            # Find zones to freeze
            zones_by_priority = sorted(
                [z for z in self.zones.values() if z.state != ZoneState.FROZEN],
                key=lambda z: z.calculate_priority()
            )
            
            # Freeze lowest priority zones
            for zone in zones_by_priority[:len(zones_by_priority)//3]:
                if zone.should_freeze():
                    await self._freeze_zone(zone)
                    
                    # Check if pressure relieved
                    memory_usage_ratio = self.used_memory_bytes / self.total_memory_bytes
                    if memory_usage_ratio < self.compression_threshold * 0.9:
                        break
    
    async def _maintenance_loop(self):
        """Background maintenance task"""
        while True:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds
                
                # Update access frequencies
                current_time = time.time()
                for zone in self.zones.values():
                    age = current_time - zone.creation_time
                    zone.access_frequency = zone.access_count / (age + 1)
                
                # Manage memory
                await self._manage_memory_pressure()
                
                # Log status
                active = sum(1 for z in self.zones.values() if z.state == ZoneState.ACTIVE)
                frozen = sum(1 for z in self.zones.values() if z.state == ZoneState.FROZEN)
                memory_used_mb = self.used_memory_bytes / (1024 * 1024)
                compressed_mb = self.compressed_memory_bytes / (1024 * 1024)
                
                logger.info(f"Memory Status: {memory_used_mb:.1f}MB used, "
                          f"{compressed_mb:.1f}MB compressed, "
                          f"{active} active zones, {frozen} frozen zones, "
                          f"cache hit rate: {self.cache_hits/(self.cache_hits+self.cache_misses+1):.1%}")
                
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current memory manager status"""
        return {
            "total_zones": len(self.zones),
            "active_zones": len(self.active_zones),
            "frozen_zones": sum(1 for z in self.zones.values() if z.state == ZoneState.FROZEN),
            "memory_used_mb": self.used_memory_bytes / (1024 * 1024),
            "memory_compressed_mb": self.compressed_memory_bytes / (1024 * 1024),
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses + 1),
            "compression_events": self.compression_events,
            "decompression_events": self.decompression_events
        }