#!/usr/bin/env python3
"""
Oracle-backed Semantic Search Evaluation
Compares AI Code Intelligence search against brute-force baseline
PASS/FAIL: Recall@10 ≥ 0.95, NDCG@10 ≥ 0.97
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
import asyncio

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tests.harness.semantic_oracles import (
    BruteForceSemanticSearch,
    SemanticSearchEvaluator, 
    SemanticQuery,
    SemanticDocument,
    SearchResult,
    SyntheticDatasetGenerator
)
from tests.harness.test_config import TestConfig, TestProfile
from ai_code_intelligence.infra.golden_bus_adapter import GoldenBusAdapter
from memory.hierarchical_memory_fixed import EnhancedSemanticSearch

class MockAICodeIntelligenceSearch:
    """Mock of AI Code Intelligence search - replace with actual import"""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.documents = {}
        
    def add_documents(self, docs: List[SemanticDocument]):
        """Add documents to search index"""
        for doc in docs:
            self.documents[doc.doc_id] = doc
    
    async def search(self, query_vector: np.ndarray, k: int = 10) -> List[SearchResult]:
        """
        Mock search with intentional quality variations
        This simulates real search with some errors
        """
        if not self.documents:
            return []
        
        # Calculate similarities (with some noise to simulate imperfect search)
        similarities = []
        
        for doc_id, doc in self.documents.items():
            # Cosine similarity with noise
            similarity = self._noisy_cosine_similarity(query_vector, doc.vector)
            similarities.append((similarity, doc_id))
        
        # Sort and take top k (may not be perfect order due to noise)
        similarities.sort(reverse=True)
        
        results = []
        for rank, (score, doc_id) in enumerate(similarities[:k]):
            results.append(SearchResult(
                doc_id=doc_id,
                score=score,
                rank=rank
            ))
        
        return results
    
    def _noisy_cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity with controlled noise"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        base_similarity = np.dot(a, b) / (norm_a * norm_b)
        
        # Add small noise (±2%) to simulate imperfect search
        noise = self.rng.uniform(-0.02, 0.02)
        return max(0.0, min(1.0, base_similarity + noise))

class SemanticEvaluationSuite:
    """Evaluate semantic search quality against oracle baseline"""
    
    def __init__(self, adapter: GoldenBusAdapter):
        self.adapter = adapter
        self.violations = []
        
        # Quality thresholds (from GAIA requirements)
        self.RECALL_THRESHOLD = 0.95
        self.NDCG_THRESHOLD = 0.97
        self.LATENCY_P99_MS = 10.0
    
    async def test_search_quality_vs_baseline(self, num_documents: int = 1000) -> Dict:
        """
        TEST: Search quality meets thresholds vs brute-force baseline
        PASS: Recall@10 ≥ 0.95 AND NDCG@10 ≥ 0.97
        """
        print(f"🧪 TEST: Search Quality vs Baseline ({num_documents} docs)")
        
        with self.adapter.span("semantic_quality_test", num_documents=num_documents):
            # Generate synthetic dataset
            generator = SyntheticDatasetGenerator(seed=42)
            documents, queries = generator.generate_clustered_dataset(
                num_documents=num_documents,
                num_clusters=10,
                vector_dim=512
            )
            
            # Setup oracle baseline
            oracle = BruteForceSemanticSearch()
            oracle.add_documents(documents)
            
            # Setup AI Code Intelligence search with enhanced implementation
            ai_search = EnhancedSemanticSearch(min_recall=0.95)
            # Convert documents to proper format
            document_vectors = [doc.vector for doc in documents]
            document_metadata = [{
                "doc_id": doc.doc_id,
                "category": doc.metadata.get("category", "") if doc.metadata else ""
            } for doc in documents]
            
            # Create evaluator
            evaluator = SemanticSearchEvaluator(oracle)
            
            # Evaluate each query
            evaluations = []
            latencies = []
            
            for query in queries[:20]:  # Test first 20 queries
                # Run AI search with enhanced implementation
                start_time = time.time()
                search_results = ai_search.search(
                    query.query_vector,
                    document_vectors,
                    document_metadata,
                    k=10
                )
                latency_ms = (time.time() - start_time) * 1000
                latencies.append(latency_ms)
                
                # Convert to SearchResult format
                ai_results = []
                for rank, (idx, score) in enumerate(search_results):
                    ai_results.append(SearchResult(
                        doc_id=documents[idx].doc_id,
                        score=score,
                        rank=rank
                    ))
                
                # Evaluate quality
                evaluation = evaluator.evaluate_search_quality(query, ai_results, k=10)
                evaluations.append(evaluation)
                
                # Record metrics
                self.adapter.record_semantic_quality(
                    recall_at_k=evaluation.recall_at_k,
                    ndcg_at_k=evaluation.ndcg_at_k,
                    k=10
                )
                
                # Check thresholds
                if evaluation.recall_at_k < self.RECALL_THRESHOLD:
                    self.violations.append({
                        "type": "recall_violation",
                        "query_id": query.query_id,
                        "recall": evaluation.recall_at_k,
                        "threshold": self.RECALL_THRESHOLD
                    })
                
                if evaluation.ndcg_at_k < self.NDCG_THRESHOLD:
                    self.violations.append({
                        "type": "ndcg_violation",
                        "query_id": query.query_id,
                        "ndcg": evaluation.ndcg_at_k,
                        "threshold": self.NDCG_THRESHOLD
                    })
            
            # Calculate aggregate metrics
            avg_recall = np.mean([e.recall_at_k for e in evaluations])
            avg_ndcg = np.mean([e.ndcg_at_k for e in evaluations])
            p99_latency = np.percentile(latencies, 99)
            
            return {
                "num_queries": len(evaluations),
                "avg_recall_at_10": avg_recall,
                "avg_ndcg_at_10": avg_ndcg,
                "p99_latency_ms": p99_latency,
                "recall_passed": avg_recall >= self.RECALL_THRESHOLD,
                "ndcg_passed": avg_ndcg >= self.NDCG_THRESHOLD,
                "latency_passed": p99_latency <= self.LATENCY_P99_MS,
                "violations": len(self.violations)
            }
    
    async def test_quality_degradation_detection(self) -> Dict:
        """
        TEST: System detects when quality drops below threshold
        PASS: Degradation detected within 2 evaluation windows
        """
        print("🧪 TEST: Quality Degradation Detection")
        
        with self.adapter.span("quality_degradation_test"):
            # Generate dataset
            generator = SyntheticDatasetGenerator(seed=42)
            base_docs, drift_docs = generator.generate_accuracy_drift_dataset(
                num_documents=500,
                vector_dim=256,
                drift_factor=0.3
            )
            
            # Setup oracle with base documents
            oracle = BruteForceSemanticSearch()
            oracle.add_documents(base_docs)
            
            # Setup AI search with drifted documents (simulating degradation)
            ai_search = EnhancedSemanticSearch(min_recall=0.95)
            drift_vectors = [doc.vector for doc in drift_docs]
            drift_metadata = [{
                "doc_id": doc.doc_id,
                "category": doc.metadata.get("category", "") if doc.metadata else ""
            } for doc in drift_docs]
            
            evaluator = SemanticSearchEvaluator(oracle)
            
            # Generate test queries
            test_queries = []
            for i in range(10):
                query_vector = generator.rng.randn(256).astype(np.float32)
                query = SemanticQuery(
                    query_id=f"drift_query_{i}",
                    query_vector=query_vector
                )
                test_queries.append(query)
            
            # Evaluate with sliding window
            window_size = 5
            windows_to_detection = None
            
            for window_idx in range(3):  # 3 windows
                window_evaluations = []
                
                for query in test_queries[window_idx:window_idx+window_size]:
                    search_results = ai_search.search(
                        query.query_vector,
                        drift_vectors,
                        drift_metadata,
                        k=10
                    )
                    ai_results = []
                    for rank, (idx, score) in enumerate(search_results):
                        ai_results.append(SearchResult(
                            doc_id=drift_docs[idx].doc_id,
                            score=score,
                            rank=rank
                        ))
                    evaluation = evaluator.evaluate_search_quality(query, ai_results, k=10)
                    window_evaluations.append(evaluation)
                
                # Calculate window average
                window_recall = np.mean([e.recall_at_k for e in window_evaluations])
                
                # Check if degradation detected
                if window_recall < self.RECALL_THRESHOLD and windows_to_detection is None:
                    windows_to_detection = window_idx + 1
                    
                    # Record detection event
                    self.adapter.record_model_switch(
                        from_model="primary",
                        to_model="fallback",
                        accuracy=window_recall,
                        threshold=self.RECALL_THRESHOLD
                    )
            
            detection_passed = windows_to_detection is not None and windows_to_detection <= 2
            
            return {
                "windows_evaluated": 3,
                "windows_to_detection": windows_to_detection or "Not detected",
                "detection_passed": detection_passed
            }
    
    async def test_search_consistency(self) -> Dict:
        """
        TEST: Search results are consistent for same query
        PASS: 100% consistency across repeated searches
        """
        print("🧪 TEST: Search Result Consistency")
        
        with self.adapter.span("search_consistency_test"):
            # Setup search with fixed seed for determinism
            generator = SyntheticDatasetGenerator(seed=42)
            documents, _ = generator.generate_clustered_dataset(
                num_documents=100,
                num_clusters=5,
                vector_dim=128
            )
            
            ai_search = EnhancedSemanticSearch(min_recall=0.95)
            doc_vectors = [doc.vector for doc in documents]
            doc_metadata = [{
                "doc_id": doc.doc_id,
                "category": doc.metadata.get("category", "") if doc.metadata else ""
            } for doc in documents]
            
            # Test query
            query_vector = generator.rng.randn(128).astype(np.float32)
            
            # Run same query multiple times
            results_sets = []
            for run in range(5):
                search_results = ai_search.search(
                    query_vector,
                    doc_vectors,
                    doc_metadata,
                    k=10
                )
                result_ids = [documents[idx].doc_id for idx, _ in search_results]
                results_sets.append(result_ids)
            
            # Check consistency
            first_result = results_sets[0]
            all_consistent = all(r == first_result for r in results_sets)
            
            if not all_consistent:
                self.violations.append({
                    "type": "consistency_violation",
                    "issue": "Same query produced different results"
                })
            
            return {
                "runs": len(results_sets),
                "consistent": all_consistent,
                "violations": len([v for v in self.violations if v["type"] == "consistency_violation"])
            }

async def run_semantic_evaluation():
    """Main evaluation runner with crisp pass/fail"""
    
    print("🔬 SEMANTIC SEARCH EVALUATION SUITE")
    print("=" * 60)
    print("REQUIREMENTS:")
    print("  • Recall@10 ≥ 0.95")
    print("  • NDCG@10 ≥ 0.97")
    print("  • P99 latency ≤ 10ms")
    print("  • Degradation detection ≤ 2 windows")
    print("=" * 60)
    
    # Create test configuration
    config = TestConfig.create(TestProfile.SMALL, seed=42)
    adapter = GoldenBusAdapter(config)
    
    # Run evaluation suite
    evaluator = SemanticEvaluationSuite(adapter)
    
    results = {}
    results["quality"] = await evaluator.test_search_quality_vs_baseline(num_documents=500)
    results["degradation"] = await evaluator.test_quality_degradation_detection()
    results["consistency"] = await evaluator.test_search_consistency()
    
    # Calculate overall pass/fail
    quality_passed = (results["quality"]["recall_passed"] and 
                     results["quality"]["ndcg_passed"] and
                     results["quality"]["latency_passed"])
    
    degradation_passed = results["degradation"]["detection_passed"]
    consistency_passed = results["consistency"]["consistent"]
    
    all_passed = quality_passed and degradation_passed and consistency_passed
    
    # Print results
    print(f"\n📊 EVALUATION RESULTS")
    print(f"Config Hash: {adapter.config_hash}")
    
    print(f"\n🎯 QUALITY METRICS:")
    print(f"   Recall@10: {results['quality']['avg_recall_at_10']:.3f} (threshold: 0.95)")
    print(f"   NDCG@10: {results['quality']['avg_ndcg_at_10']:.3f} (threshold: 0.97)")
    print(f"   P99 Latency: {results['quality']['p99_latency_ms']:.2f}ms (threshold: 10ms)")
    
    quality_status = "✅ PASS" if quality_passed else "❌ FAIL"
    print(f"   {quality_status} Quality Requirements")
    
    degradation_status = "✅ PASS" if degradation_passed else "❌ FAIL"
    print(f"\n   {degradation_status} Degradation Detection: {results['degradation']['windows_to_detection']} windows")
    
    consistency_status = "✅ PASS" if consistency_passed else "❌ FAIL"
    print(f"   {consistency_status} Search Consistency")
    
    # Overall verdict
    print(f"\n🏆 OVERALL VERDICT: {'✅ PASS' if all_passed else '❌ FAIL'}")
    
    if all_passed:
        print("   ✅ Search quality meets all requirements")
        print("   ✅ System ready for production")
    else:
        print(f"   ❌ {len(evaluator.violations)} violations found")
        
        if not quality_passed:
            print("   ❌ Search quality below thresholds")
        if not degradation_passed:
            print("   ❌ Degradation detection too slow")
        if not consistency_passed:
            print("   ❌ Search results not consistent")
    
    # Export telemetry
    adapter.export_telemetry(f"semantic_eval_{adapter.config_hash}.json")
    
    return all_passed

if __name__ == "__main__":
    passed = asyncio.run(run_semantic_evaluation())
    exit(0 if passed else 1)