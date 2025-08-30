#!/usr/bin/env python3
"""
Semantic Search Oracles and Baselines - Ground truth for testing semantic search quality
"""

import numpy as np
import time
import hashlib
import json
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, asdict
import heapq
from collections import defaultdict
import math

@dataclass
class SemanticQuery:
    """A semantic search query with expected results"""
    query_id: str
    query_vector: np.ndarray
    query_text: Optional[str] = None
    expected_results: Optional[List[str]] = None  # Known relevant document IDs
    relevance_scores: Optional[Dict[str, float]] = None  # Manual relevance scores

@dataclass
class SemanticDocument:
    """A document in the semantic search corpus"""
    doc_id: str
    vector: np.ndarray
    text: Optional[str] = None
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SearchResult:
    """A single search result"""
    doc_id: str
    score: float
    rank: int
    
@dataclass
class SearchEvaluation:
    """Evaluation metrics for semantic search results"""
    query_id: str
    k: int
    recall_at_k: float
    precision_at_k: float
    ndcg_at_k: float
    mrr: float  # Mean Reciprocal Rank
    latency_ms: float
    num_candidates: int
    
class BruteForceSemanticSearch:
    """Brute force semantic search baseline for correctness validation"""
    
    def __init__(self):
        self.documents: Dict[str, SemanticDocument] = {}
        self.index_version = 0
    
    def add_document(self, doc: SemanticDocument):
        """Add document to the index"""
        self.documents[doc.doc_id] = doc
        self.index_version += 1
    
    def add_documents(self, docs: List[SemanticDocument]):
        """Add multiple documents to the index"""
        for doc in docs:
            self.documents[doc.doc_id] = doc
        self.index_version += 1
    
    def search(self, query_vector: np.ndarray, k: int = 10, 
               similarity_threshold: float = -np.inf) -> List[SearchResult]:
        """Brute force exact nearest neighbor search"""
        if len(self.documents) == 0:
            return []
        
        # Calculate similarities with all documents
        similarities = []
        
        for doc_id, doc in self.documents.items():
            # Cosine similarity
            similarity = self._cosine_similarity(query_vector, doc.vector)
            
            if similarity >= similarity_threshold:
                similarities.append((similarity, doc_id))
        
        # Sort by similarity (descending) and take top k
        similarities.sort(reverse=True)
        
        results = []
        for rank, (score, doc_id) in enumerate(similarities[:k]):
            results.append(SearchResult(
                doc_id=doc_id,
                score=score,
                rank=rank
            ))
        
        return results
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return np.dot(a, b) / (norm_a * norm_b)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        if not self.documents:
            return {"num_documents": 0, "index_version": self.index_version}
        
        vector_dims = [len(doc.vector) for doc in self.documents.values()]
        categories = [doc.category for doc in self.documents.values() if doc.category]
        
        return {
            "num_documents": len(self.documents),
            "index_version": self.index_version,
            "vector_dimensions": {
                "min": min(vector_dims) if vector_dims else 0,
                "max": max(vector_dims) if vector_dims else 0,
                "avg": sum(vector_dims) / len(vector_dims) if vector_dims else 0
            },
            "categories": len(set(categories)) if categories else 0
        }

class SemanticSearchEvaluator:
    """Evaluates semantic search quality against ground truth"""
    
    def __init__(self, baseline_oracle: BruteForceSemanticSearch):
        self.baseline_oracle = baseline_oracle
        self.evaluation_cache: Dict[str, SearchEvaluation] = {}
    
    def evaluate_search_quality(self, 
                               query: SemanticQuery,
                               test_results: List[SearchResult],
                               k: int = 10) -> SearchEvaluation:
        """Compare test results against baseline oracle"""
        
        # Generate cache key
        cache_key = self._generate_cache_key(query, test_results, k)
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        start_time = time.time()
        
        # Get ground truth from baseline oracle
        baseline_results = self.baseline_oracle.search(query.query_vector, k=k)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Calculate evaluation metrics
        evaluation = self._calculate_metrics(
            query, test_results, baseline_results, k, latency_ms
        )
        
        # Cache result
        self.evaluation_cache[cache_key] = evaluation
        
        return evaluation
    
    def _generate_cache_key(self, query: SemanticQuery, results: List[SearchResult], k: int) -> str:
        """Generate cache key for evaluation result"""
        # Create a hash of query vector and results
        query_hash = hashlib.md5(query.query_vector.tobytes()).hexdigest()[:8]
        results_str = ",".join([f"{r.doc_id}:{r.score:.4f}" for r in results[:k]])
        results_hash = hashlib.md5(results_str.encode()).hexdigest()[:8]
        
        return f"{query.query_id}_{query_hash}_{results_hash}_{k}"
    
    def _calculate_metrics(self,
                          query: SemanticQuery,
                          test_results: List[SearchResult],
                          baseline_results: List[SearchResult],
                          k: int,
                          latency_ms: float) -> SearchEvaluation:
        """Calculate comprehensive evaluation metrics"""
        
        # Convert to sets for easier comparison
        test_doc_ids = set(r.doc_id for r in test_results[:k])
        baseline_doc_ids = set(r.doc_id for r in baseline_results[:k])
        
        # Recall@k: What fraction of relevant docs were retrieved?
        relevant_retrieved = len(test_doc_ids.intersection(baseline_doc_ids))
        recall_at_k = relevant_retrieved / len(baseline_doc_ids) if baseline_doc_ids else 0.0
        
        # Precision@k: What fraction of retrieved docs are relevant?
        precision_at_k = relevant_retrieved / len(test_doc_ids) if test_doc_ids else 0.0
        
        # NDCG@k: Normalized Discounted Cumulative Gain
        ndcg_at_k = self._calculate_ndcg(test_results[:k], baseline_results[:k], k)
        
        # MRR: Mean Reciprocal Rank - position of first relevant result
        mrr = self._calculate_mrr(test_results[:k], baseline_doc_ids)
        
        return SearchEvaluation(
            query_id=query.query_id,
            k=k,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            ndcg_at_k=ndcg_at_k,
            mrr=mrr,
            latency_ms=latency_ms,
            num_candidates=len(test_results)
        )
    
    def _calculate_ndcg(self, test_results: List[SearchResult], 
                       baseline_results: List[SearchResult], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@k"""
        
        if not baseline_results:
            return 0.0
        
        # Create relevance mapping from baseline (perfect ranking)
        relevance_map = {}
        for i, result in enumerate(baseline_results):
            # Use inverse rank as relevance score (1.0 for rank 0, 0.5 for rank 1, etc.)
            relevance_map[result.doc_id] = 1.0 / (i + 1)
        
        # Calculate DCG for test results
        dcg = 0.0
        for i, result in enumerate(test_results[:k]):
            relevance = relevance_map.get(result.doc_id, 0.0)
            if i == 0:
                dcg += relevance
            else:
                dcg += relevance / math.log2(i + 1)
        
        # Calculate IDCG (perfect ranking)
        idcg = 0.0
        sorted_relevances = sorted(relevance_map.values(), reverse=True)
        for i, relevance in enumerate(sorted_relevances[:k]):
            if i == 0:
                idcg += relevance
            else:
                idcg += relevance / math.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_mrr(self, test_results: List[SearchResult], 
                      relevant_doc_ids: Set[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, result in enumerate(test_results):
            if result.doc_id in relevant_doc_ids:
                return 1.0 / (i + 1)
        return 0.0
    
    def batch_evaluate(self, queries_and_results: List[Tuple[SemanticQuery, List[SearchResult]]], 
                      k: int = 10) -> Dict[str, Any]:
        """Evaluate multiple queries and return aggregate statistics"""
        
        evaluations = []
        for query, results in queries_and_results:
            evaluation = self.evaluate_search_quality(query, results, k)
            evaluations.append(evaluation)
        
        if not evaluations:
            return {"error": "No evaluations performed"}
        
        # Calculate aggregate metrics
        recall_values = [e.recall_at_k for e in evaluations]
        precision_values = [e.precision_at_k for e in evaluations]
        ndcg_values = [e.ndcg_at_k for e in evaluations]
        mrr_values = [e.mrr for e in evaluations]
        latency_values = [e.latency_ms for e in evaluations]
        
        return {
            "num_queries": len(evaluations),
            "k": k,
            "metrics": {
                "recall_at_k": {
                    "mean": np.mean(recall_values),
                    "std": np.std(recall_values),
                    "min": np.min(recall_values),
                    "max": np.max(recall_values),
                    "p95": np.percentile(recall_values, 95)
                },
                "precision_at_k": {
                    "mean": np.mean(precision_values),
                    "std": np.std(precision_values),
                    "min": np.min(precision_values),
                    "max": np.max(precision_values),
                    "p95": np.percentile(precision_values, 95)
                },
                "ndcg_at_k": {
                    "mean": np.mean(ndcg_values),
                    "std": np.std(ndcg_values),
                    "min": np.min(ndcg_values),
                    "max": np.max(ndcg_values),
                    "p95": np.percentile(ndcg_values, 95)
                },
                "mrr": {
                    "mean": np.mean(mrr_values),
                    "std": np.std(mrr_values),
                    "min": np.min(mrr_values),
                    "max": np.max(mrr_values),
                    "p95": np.percentile(mrr_values, 95)
                },
                "latency_ms": {
                    "mean": np.mean(latency_values),
                    "std": np.std(latency_values),
                    "min": np.min(latency_values),
                    "max": np.max(latency_values),
                    "p95": np.percentile(latency_values, 95),
                    "p99": np.percentile(latency_values, 99)
                }
            },
            "individual_evaluations": [asdict(e) for e in evaluations]
        }

class SyntheticDatasetGenerator:
    """Generate synthetic semantic search datasets for testing"""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.seed = seed
    
    def generate_clustered_dataset(self, 
                                 num_documents: int = 1000,
                                 num_clusters: int = 10,
                                 vector_dim: int = 512,
                                 cluster_std: float = 0.1) -> Tuple[List[SemanticDocument], List[SemanticQuery]]:
        """Generate documents clustered in semantic space with queries"""
        
        # Generate cluster centers
        cluster_centers = self.rng.randn(num_clusters, vector_dim)
        cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
        
        documents = []
        queries = []
        
        # Assign documents to clusters
        docs_per_cluster = num_documents // num_clusters
        
        for cluster_id in range(num_clusters):
            cluster_center = cluster_centers[cluster_id]
            
            # Generate documents in this cluster
            cluster_docs = []
            for doc_idx in range(docs_per_cluster):
                # Add noise around cluster center
                noise = self.rng.normal(0, cluster_std, vector_dim)
                doc_vector = cluster_center + noise
                doc_vector = doc_vector / np.linalg.norm(doc_vector)  # Normalize
                
                doc_id = f"doc_{cluster_id}_{doc_idx}"
                doc = SemanticDocument(
                    doc_id=doc_id,
                    vector=doc_vector.astype(np.float32),
                    category=f"cluster_{cluster_id}",
                    text=f"Document {doc_idx} in cluster {cluster_id}",
                    metadata={"cluster_id": cluster_id, "doc_index": doc_idx}
                )
                
                documents.append(doc)
                cluster_docs.append(doc)
            
            # Generate queries for this cluster
            num_queries_per_cluster = max(1, num_clusters // 5)  # 20% of clusters get queries
            
            if cluster_id < num_queries_per_cluster:
                # Query vector close to cluster center
                query_noise = self.rng.normal(0, cluster_std * 0.5, vector_dim)
                query_vector = cluster_center + query_noise
                query_vector = query_vector / np.linalg.norm(query_vector)
                
                # Expected results are documents in the same cluster
                expected_results = [doc.doc_id for doc in cluster_docs[:10]]  # Top 10 in cluster
                
                query = SemanticQuery(
                    query_id=f"query_{cluster_id}",
                    query_vector=query_vector.astype(np.float32),
                    query_text=f"Query for cluster {cluster_id}",
                    expected_results=expected_results
                )
                
                queries.append(query)
        
        return documents, queries
    
    def generate_accuracy_drift_dataset(self,
                                       num_documents: int = 500,
                                       vector_dim: int = 256,
                                       drift_factor: float = 0.3) -> Tuple[List[SemanticDocument], List[SemanticDocument]]:
        """Generate two datasets with controlled drift for accuracy testing"""
        
        # Generate base dataset
        base_vectors = self.rng.randn(num_documents, vector_dim).astype(np.float32)
        base_vectors = base_vectors / np.linalg.norm(base_vectors, axis=1, keepdims=True)
        
        base_docs = []
        for i, vector in enumerate(base_vectors):
            doc = SemanticDocument(
                doc_id=f"base_doc_{i}",
                vector=vector,
                category="base",
                text=f"Base document {i}"
            )
            base_docs.append(doc)
        
        # Generate drifted dataset
        drift_noise = self.rng.normal(0, drift_factor, (num_documents, vector_dim))
        drifted_vectors = base_vectors + drift_noise.astype(np.float32)
        drifted_vectors = drifted_vectors / np.linalg.norm(drifted_vectors, axis=1, keepdims=True)
        
        drifted_docs = []
        for i, vector in enumerate(drifted_vectors):
            doc = SemanticDocument(
                doc_id=f"drift_doc_{i}",
                vector=vector,
                category="drifted", 
                text=f"Drifted document {i}"
            )
            drifted_docs.append(doc)
        
        return base_docs, drifted_docs
    
    def save_dataset(self, documents: List[SemanticDocument], queries: List[SemanticQuery], 
                    filename: str):
        """Save dataset to JSON file"""
        dataset = {
            "metadata": {
                "num_documents": len(documents),
                "num_queries": len(queries),
                "generator_seed": self.seed,
                "generated_at": time.time()
            },
            "documents": [],
            "queries": []
        }
        
        # Serialize documents (convert numpy arrays to lists)
        for doc in documents:
            doc_dict = asdict(doc)
            doc_dict["vector"] = doc.vector.tolist()
            dataset["documents"].append(doc_dict)
        
        # Serialize queries
        for query in queries:
            query_dict = asdict(query)
            query_dict["query_vector"] = query.query_vector.tolist()
            dataset["queries"].append(query_dict)
        
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"💾 Dataset saved to {filename}")
    
    def load_dataset(self, filename: str) -> Tuple[List[SemanticDocument], List[SemanticQuery]]:
        """Load dataset from JSON file"""
        with open(filename, 'r') as f:
            dataset = json.load(f)
        
        documents = []
        for doc_dict in dataset["documents"]:
            doc_dict["vector"] = np.array(doc_dict["vector"], dtype=np.float32)
            doc = SemanticDocument(**doc_dict)
            documents.append(doc)
        
        queries = []
        for query_dict in dataset["queries"]:
            query_dict["query_vector"] = np.array(query_dict["query_vector"], dtype=np.float32)
            query = SemanticQuery(**query_dict)
            queries.append(query)
        
        print(f"📁 Loaded dataset: {len(documents)} docs, {len(queries)} queries")
        return documents, queries

# Convenience functions for testing
def create_test_oracle(num_documents: int = 1000, seed: int = 42) -> Tuple[BruteForceSemanticSearch, List[SemanticQuery]]:
    """Create a test oracle with synthetic data"""
    generator = SyntheticDatasetGenerator(seed=seed)
    documents, queries = generator.generate_clustered_dataset(
        num_documents=num_documents,
        num_clusters=max(10, num_documents // 100),
        vector_dim=512
    )
    
    oracle = BruteForceSemanticSearch()
    oracle.add_documents(documents)
    
    return oracle, queries

def run_baseline_benchmark(oracle: BruteForceSemanticSearch, queries: List[SemanticQuery], 
                          k: int = 10) -> Dict[str, Any]:
    """Benchmark the baseline oracle performance"""
    latencies = []
    
    for query in queries:
        start_time = time.time()
        results = oracle.search(query.query_vector, k=k)
        latency_ms = (time.time() - start_time) * 1000
        latencies.append(latency_ms)
    
    return {
        "num_queries": len(queries),
        "k": k,
        "index_stats": oracle.get_index_stats(),
        "latency_ms": {
            "mean": np.mean(latencies),
            "std": np.std(latencies),
            "min": np.min(latencies),
            "max": np.max(latencies),
            "p50": np.percentile(latencies, 50),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99)
        }
    }