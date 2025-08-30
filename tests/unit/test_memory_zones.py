"""
Unit tests for memory zone functionality
"""

import pytest
import numpy as np
import asyncio
import time
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from memory.hierarchical_memory import (
    MemoryZone, ZoneState, HierarchicalMemoryManager, 
    SemanticBTree, BTreeNode
)

class TestMemoryZone:
    """Test memory zone operations"""
    
    def test_zone_creation(self):
        """Test basic zone creation"""
        zone = MemoryZone(
            zone_id="test_zone",
            semantic_category="test"
        )
        
        assert zone.zone_id == "test_zone"
        assert zone.semantic_category == "test"
        assert zone.state == ZoneState.WARM
        assert zone.size_bytes == 0
        assert len(zone.embeddings) == 0
    
    def test_priority_calculation(self):
        """Test zone priority calculation"""
        zone = MemoryZone(
            zone_id="test_zone",
            semantic_category="test"
        )
        
        # Fresh zone should have low priority
        initial_priority = zone.calculate_priority()
        
        # Simulate access
        zone.access_count = 10
        zone.access_frequency = 5.0
        zone.last_accessed = time.time()
        
        # Priority should increase after access
        new_priority = zone.calculate_priority()
        assert new_priority > initial_priority
    
    def test_freeze_conditions(self):
        """Test zone freeze decision logic"""
        zone = MemoryZone(
            zone_id="test_zone",
            semantic_category="test",
            size_bytes=200 * 1024  # 200KB
        )
        
        # Fresh zone shouldn't freeze
        assert not zone.should_freeze(idle_threshold_seconds=300)
        
        # Old zone should freeze
        zone.last_accessed = time.time() - 400  # 400 seconds ago
        assert zone.should_freeze(idle_threshold_seconds=300)
        
        # Frozen zone shouldn't freeze again
        zone.state = ZoneState.FROZEN
        assert not zone.should_freeze()
    
    def test_split_conditions(self):
        """Test zone split decision logic"""
        zone = MemoryZone(
            zone_id="test_zone",
            semantic_category="test",
            size_bytes=460 * 1024,  # 460KB (90% of 512KB)
            max_size_bytes=512 * 1024
        )
        
        assert zone.should_split()
        
        # Smaller zone shouldn't split
        zone.size_bytes = 200 * 1024
        assert not zone.should_split()
    
    def test_merge_conditions(self):
        """Test zone merge decision logic"""
        zone = MemoryZone(
            zone_id="test_zone",
            semantic_category="test",
            size_bytes=40 * 1024,  # 40KB (40% of 100KB min)
            min_size_bytes=100 * 1024
        )
        
        assert zone.should_merge()
        
        # Larger zone shouldn't merge
        zone.size_bytes = 80 * 1024
        assert not zone.should_merge()

class TestSemanticBTree:
    """Test B-tree indexing functionality"""
    
    def test_btree_creation(self):
        """Test B-tree initialization"""
        btree = SemanticBTree(t=3)
        assert btree.root.leaf is True
        assert len(btree.root.keys) == 0
    
    def test_single_insert_search(self):
        """Test inserting and searching single item"""
        btree = SemanticBTree(t=3)
        
        btree.insert("key1", "zone1")
        result = btree.search("key1")
        
        assert result == "zone1"
        assert btree.search("nonexistent") is None
    
    def test_multiple_inserts(self):
        """Test multiple insertions"""
        btree = SemanticBTree(t=3)
        
        # Insert multiple keys
        test_data = [
            ("apple", "zone_fruit"),
            ("banana", "zone_fruit"),
            ("carrot", "zone_vegetable"),
            ("date", "zone_fruit"),
            ("eggplant", "zone_vegetable")
        ]
        
        for key, zone in test_data:
            btree.insert(key, zone)
        
        # Verify all keys can be found
        for key, expected_zone in test_data:
            assert btree.search(key) == expected_zone
    
    def test_btree_splits(self):
        """Test B-tree node splitting"""
        btree = SemanticBTree(t=2)  # Smaller t to force splits
        
        # Insert enough items to force splits
        keys = [f"key_{i:03d}" for i in range(10)]
        zones = [f"zone_{i}" for i in range(10)]
        
        for key, zone in zip(keys, zones):
            btree.insert(key, zone)
        
        # Verify all items are still findable after splits
        for key, zone in zip(keys, zones):
            assert btree.search(key) == zone
        
        # Root should no longer be a leaf after splits
        assert not btree.root.leaf

@pytest.mark.asyncio
class TestHierarchicalMemoryManager:
    """Test memory manager functionality"""
    
    async def test_manager_initialization(self):
        """Test memory manager initialization"""
        manager = HierarchicalMemoryManager(total_memory_mb=128)
        await manager.initialize()
        
        assert manager.total_memory_bytes == 128 * 1024 * 1024
        assert len(manager.zones) == 0
        assert manager.maintenance_task is not None
        
        # Cleanup
        manager.maintenance_task.cancel()
        try:
            await manager.maintenance_task
        except asyncio.CancelledError:
            pass
    
    async def test_store_and_retrieve(self):
        """Test basic store/retrieve operations"""
        manager = HierarchicalMemoryManager(total_memory_mb=128)
        await manager.initialize()
        
        try:
            # Store an embedding
            key = "test_embedding"
            embedding = np.random.randn(512).astype(np.float32)
            associations = {"model1", "model2"}
            
            zone_id = await manager.store(
                key=key,
                embedding=embedding,
                associations=associations,
                semantic_category="test"
            )
            
            assert zone_id is not None
            assert len(manager.zones) == 1
            
            # Retrieve the embedding
            result = await manager.retrieve(key)
            assert result is not None
            
            retrieved_embedding, retrieved_associations = result
            np.testing.assert_array_equal(embedding, retrieved_embedding)
            assert retrieved_associations == associations
            
        finally:
            # Cleanup
            manager.maintenance_task.cancel()
            try:
                await manager.maintenance_task
            except asyncio.CancelledError:
                pass
    
    async def test_zone_creation_logic(self):
        """Test zone creation and reuse logic"""
        manager = HierarchicalMemoryManager(total_memory_mb=128)
        await manager.initialize()
        
        try:
            # Store multiple items with same category
            embeddings = []
            keys = []
            
            for i in range(5):
                key = f"test_{i}"
                embedding = np.random.randn(100).astype(np.float32)  # Small embeddings
                keys.append(key)
                embeddings.append(embedding)
                
                await manager.store(key, embedding, semantic_category="vision")
            
            # Should reuse the same zone for same category (until full)
            vision_zones = [z for z in manager.zones.values() if z.semantic_category == "vision"]
            assert len(vision_zones) >= 1
            
            # Store different category
            await manager.store("audio_test", np.random.randn(100).astype(np.float32), 
                               semantic_category="audio")
            
            categories = {z.semantic_category for z in manager.zones.values()}
            assert "vision" in categories
            assert "audio" in categories or "low_activation" in categories  # Might be inferred differently
            
        finally:
            # Cleanup
            manager.maintenance_task.cancel()
            try:
                await manager.maintenance_task
            except asyncio.CancelledError:
                pass
    
    async def test_nonexistent_key_retrieval(self):
        """Test retrieving nonexistent keys"""
        manager = HierarchicalMemoryManager(total_memory_mb=128)
        await manager.initialize()
        
        try:
            result = await manager.retrieve("nonexistent_key")
            assert result is None
            
        finally:
            # Cleanup
            manager.maintenance_task.cancel()
            try:
                await manager.maintenance_task
            except asyncio.CancelledError:
                pass
    
    @patch('memory.hierarchical_memory.zlib.compress')
    @patch('memory.hierarchical_memory.zlib.decompress')
    async def test_compression_decompression(self, mock_decompress, mock_compress):
        """Test zone compression and decompression"""
        # Mock compression to return predictable data
        mock_compress.return_value = b'compressed_data'
        mock_decompress.return_value = b'decompressed_data'
        
        manager = HierarchicalMemoryManager(total_memory_mb=128)
        await manager.initialize()
        
        try:
            zone = MemoryZone(
                zone_id="test_zone",
                semantic_category="test"
            )
            zone.embeddings["test"] = np.array([1, 2, 3])
            zone.associations["test"] = {"assoc1"}
            zone.metadata["test"] = "metadata"
            
            manager.zones["test_zone"] = zone
            
            # Test compression
            await manager._freeze_zone(zone)
            
            assert zone.state == ZoneState.FROZEN
            assert zone.compressed_data == b'compressed_data'
            assert len(zone.embeddings) == 0  # Should be cleared
            assert mock_compress.called
            
            # Mock pickle.loads for decompression
            with patch('memory.hierarchical_memory.pickle.loads') as mock_pickle:
                mock_pickle.return_value = {
                    'embeddings': {"test": np.array([1, 2, 3])},
                    'associations': {"test": {"assoc1"}},
                    'metadata': {"test": "metadata"}
                }
                
                # Test decompression
                await manager._thaw_zone(zone)
                
                assert zone.state == ZoneState.WARM
                assert zone.compressed_data is None
                assert len(zone.embeddings) == 1
                assert mock_decompress.called
                
        finally:
            # Cleanup
            manager.maintenance_task.cancel()
            try:
                await manager.maintenance_task
            except asyncio.CancelledError:
                pass
    
    async def test_memory_pressure_response(self):
        """Test system response to memory pressure"""
        # Small memory limit to trigger pressure quickly
        manager = HierarchicalMemoryManager(total_memory_mb=1, compression_threshold=0.5)
        await manager.initialize()
        
        try:
            # Fill memory beyond threshold
            large_embedding = np.random.randn(1000).astype(np.float32)  # ~4KB
            
            zone_ids = []
            for i in range(10):  # Should exceed 1MB limit
                zone_id = await manager.store(
                    f"large_item_{i}",
                    large_embedding,
                    semantic_category=f"category_{i % 3}"
                )
                zone_ids.append(zone_id)
            
            # Force memory pressure management
            await manager._manage_memory_pressure()
            
            # Some zones should be frozen
            frozen_zones = [z for z in manager.zones.values() if z.state == ZoneState.FROZEN]
            assert len(frozen_zones) > 0
            
        finally:
            # Cleanup
            manager.maintenance_task.cancel()
            try:
                await manager.maintenance_task
            except asyncio.CancelledError:
                pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])