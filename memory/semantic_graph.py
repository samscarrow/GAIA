"""
Semantic Graph - Models associate through meaning, not hierarchy
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class SemanticNode:
    """A node in the semantic graph representing a model or concept"""
    node_id: str
    node_type: str  # 'model', 'concept', 'memory'
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(128))
    metadata: Dict[str, Any] = field(default_factory=dict)
    activation_level: float = 0.0
    last_activated: float = field(default_factory=time.time)
    access_count: int = 0
    
    def activate(self, strength: float = 1.0):
        """Activate this node"""
        self.activation_level = min(1.0, self.activation_level + strength)
        self.last_activated = time.time()
        self.access_count += 1
        
    def decay(self, rate: float = 0.1):
        """Decay activation over time"""
        time_delta = time.time() - self.last_activated
        self.activation_level *= (1.0 - rate * time_delta)
        self.activation_level = max(0.0, self.activation_level)


class SemanticGraph:
    """
    A graph structure where models and concepts associate through semantic similarity
    """
    
    def __init__(self, embedding_dim: int = 128):
        self.graph = nx.Graph()
        self.nodes: Dict[str, SemanticNode] = {}
        self.embedding_dim = embedding_dim
        self.association_threshold = 0.5
        self.temporal_associations: List[Tuple[str, str, float]] = []
        
    def add_node(self, node_id: str, node_type: str = 'model',
                 embedding: Optional[np.ndarray] = None,
                 metadata: Optional[Dict] = None) -> SemanticNode:
        """Add a node to the semantic graph"""
        if embedding is None:
            embedding = np.random.randn(self.embedding_dim)
            embedding = embedding / np.linalg.norm(embedding)
            
        node = SemanticNode(
            node_id=node_id,
            node_type=node_type,
            embedding=embedding,
            metadata=metadata or {}
        )
        
        self.nodes[node_id] = node
        self.graph.add_node(node_id, data=node)
        
        # Automatically create associations based on semantic similarity
        self._update_associations(node_id)
        
        logger.info(f"Added node {node_id} of type {node_type}")
        return node
        
    def _update_associations(self, node_id: str):
        """Update associations based on semantic similarity"""
        node = self.nodes[node_id]
        
        for other_id, other_node in self.nodes.items():
            if other_id == node_id:
                continue
                
            # Calculate semantic similarity
            similarity = self._calculate_similarity(node.embedding, other_node.embedding)
            
            if similarity > self.association_threshold:
                # Create or update edge
                if self.graph.has_edge(node_id, other_id):
                    # Update weight
                    current_weight = self.graph[node_id][other_id]['weight']
                    new_weight = (current_weight + similarity) / 2
                    self.graph[node_id][other_id]['weight'] = new_weight
                else:
                    # Create new edge
                    self.graph.add_edge(node_id, other_id, weight=similarity)
                    logger.debug(f"Created association: {node_id} <-> {other_id} (weight: {similarity:.3f})")
                    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
        
    def activate_node(self, node_id: str, strength: float = 1.0,
                     propagate: bool = True, depth: int = 2):
        """Activate a node and optionally propagate to neighbors"""
        if node_id not in self.nodes:
            return
            
        node = self.nodes[node_id]
        node.activate(strength)
        
        if propagate and depth > 0:
            # Propagate activation to neighbors
            neighbors = self.graph.neighbors(node_id)
            for neighbor_id in neighbors:
                edge_weight = self.graph[node_id][neighbor_id]['weight']
                propagated_strength = strength * edge_weight * 0.5  # Decay
                self.activate_node(neighbor_id, propagated_strength, True, depth - 1)
                
    def find_associations(self, node_id: str, min_weight: float = 0.3,
                         max_distance: int = 2) -> List[Tuple[str, float]]:
        """Find associated nodes within a certain distance"""
        if node_id not in self.graph:
            return []
            
        associations = []
        
        # Use Dijkstra to find shortest paths
        lengths = nx.single_source_dijkstra_path_length(
            self.graph, node_id, cutoff=max_distance, weight='weight'
        )
        
        for target_id, distance in lengths.items():
            if target_id == node_id:
                continue
                
            # Get direct edge weight if exists
            if self.graph.has_edge(node_id, target_id):
                weight = self.graph[node_id][target_id]['weight']
                if weight >= min_weight:
                    associations.append((target_id, weight))
                    
        return sorted(associations, key=lambda x: x[1], reverse=True)
        
    def create_temporal_association(self, node1: str, node2: str):
        """Create association based on temporal co-occurrence"""
        timestamp = time.time()
        self.temporal_associations.append((node1, node2, timestamp))
        
        # Update edge weight based on temporal association
        if self.graph.has_edge(node1, node2):
            current_weight = self.graph[node1][node2]['weight']
            # Increase weight slightly for temporal association
            new_weight = min(1.0, current_weight + 0.1)
            self.graph[node1][node2]['weight'] = new_weight
        else:
            self.graph.add_edge(node1, node2, weight=0.3)
            
    def get_activation_pattern(self) -> Dict[str, float]:
        """Get current activation pattern across all nodes"""
        return {
            node_id: node.activation_level
            for node_id, node in self.nodes.items()
        }
        
    def decay_all_activations(self, rate: float = 0.01):
        """Decay all node activations"""
        for node in self.nodes.values():
            node.decay(rate)
            
    def find_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find associative path between two nodes"""
        if source not in self.graph or target not in self.graph:
            return None
            
        try:
            path = nx.shortest_path(self.graph, source, target, weight='weight')
            return path
        except nx.NetworkXNoPath:
            return None
            
    def get_subgraph(self, center_node: str, radius: int = 2) -> nx.Graph:
        """Get subgraph around a central node"""
        if center_node not in self.graph:
            return nx.Graph()
            
        # Get all nodes within radius
        subgraph_nodes = set([center_node])
        current_layer = set([center_node])
        
        for _ in range(radius):
            next_layer = set()
            for node in current_layer:
                next_layer.update(self.graph.neighbors(node))
            subgraph_nodes.update(next_layer)
            current_layer = next_layer
            
        return self.graph.subgraph(subgraph_nodes)
        
    def merge_embeddings(self, node_ids: List[str]) -> np.ndarray:
        """Merge embeddings from multiple nodes"""
        if not node_ids:
            return np.zeros(self.embedding_dim)
            
        embeddings = []
        weights = []
        
        for node_id in node_ids:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                embeddings.append(node.embedding)
                weights.append(node.activation_level + 0.1)  # Avoid zero weight
                
        if not embeddings:
            return np.zeros(self.embedding_dim)
            
        # Weighted average
        embeddings = np.array(embeddings)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        merged = np.average(embeddings, axis=0, weights=weights)
        # Normalize
        merged = merged / np.linalg.norm(merged)
        
        return merged
        
    def save_graph(self, filepath: str):
        """Save graph to file"""
        graph_data = {
            'nodes': {
                node_id: {
                    'type': node.node_type,
                    'embedding': node.embedding.tolist(),
                    'metadata': node.metadata,
                    'activation': node.activation_level
                }
                for node_id, node in self.nodes.items()
            },
            'edges': [
                (u, v, self.graph[u][v]['weight'])
                for u, v in self.graph.edges()
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2)
            
    def visualize_activation(self) -> Dict[str, Any]:
        """Get visualization data for current activation state"""
        return {
            'nodes': [
                {
                    'id': node_id,
                    'activation': node.activation_level,
                    'type': node.node_type,
                    'size': 10 + node.activation_level * 40  # Visual size
                }
                for node_id, node in self.nodes.items()
            ],
            'edges': [
                {
                    'source': u,
                    'target': v,
                    'weight': self.graph[u][v]['weight']
                }
                for u, v in self.graph.edges()
            ]
        }