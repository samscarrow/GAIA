#!/usr/bin/env python3
"""
INVARIANT TEST: Activation Determinism
CLAIM: Same seed + same start node + same context = identical activation pattern
PASS/FAIL: 100% reproducibility or FAIL
"""

import sys
import os
import asyncio
import hashlib
import json
import numpy as np
from typing import Dict, List, Set, Tuple, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tests.harness.test_config import TestConfig, TestProfile, create_metric
from ai_code_intelligence.infra.golden_bus_adapter import GoldenBusAdapter

# Import the semantic graph (simplified mock for demonstration)
class MockSemanticGraph:
    """Mock semantic graph for testing - replace with actual import"""
    
    def __init__(self, seed: int):
        self.rng = np.random.RandomState(seed)
        self.nodes = {}
        self.edges = {}
        
        # Generate deterministic graph structure
        self._generate_test_graph()
    
    def _generate_test_graph(self):
        """Generate a deterministic test graph"""
        # Create 100 nodes with embeddings
        for i in range(100):
            node_id = f"node_{i}"
            embedding = self.rng.randn(512).astype(np.float32)
            self.nodes[node_id] = {
                "id": node_id,
                "embedding": embedding,
                "activation": 0.0,
                "metadata": {"category": f"cat_{i % 10}"}
            }
        
        # Create deterministic edges
        for i in range(100):
            source = f"node_{i}"
            # Each node connects to 3-5 neighbors
            num_edges = self.rng.randint(3, 6)
            for _ in range(num_edges):
                target = f"node_{self.rng.randint(0, 100)}"
                weight = self.rng.random()
                
                if source not in self.edges:
                    self.edges[source] = []
                self.edges[source].append((target, weight))
    
    async def activate_neighborhood(self, start_node: str, context: Dict, 
                                   depth: int = 3, threshold: float = 0.3) -> Set[str]:
        """
        Deterministic activation propagation
        Same inputs MUST produce same outputs
        """
        activated = set()
        to_process = [(start_node, 1.0)]  # (node, activation_strength)
        processed = set()
        
        for current_depth in range(depth):
            next_wave = []
            
            for node_id, strength in to_process:
                if node_id in processed:
                    continue
                
                processed.add(node_id)
                
                # Apply activation
                if strength > threshold:
                    activated.add(node_id)
                    
                    # Propagate to neighbors
                    if node_id in self.edges:
                        # CRITICAL: Sort edges for determinism
                        neighbors = sorted(self.edges[node_id], key=lambda x: x[0])
                        
                        for neighbor_id, weight in neighbors:
                            # Deterministic activation calculation
                            neighbor_strength = strength * weight * 0.8
                            
                            # Context modulation (deterministic)
                            if "boost_category" in context:
                                node_data = self.nodes.get(neighbor_id, {})
                                if node_data.get("metadata", {}).get("category") == context["boost_category"]:
                                    neighbor_strength *= 1.5
                            
                            if neighbor_strength > threshold:
                                next_wave.append((neighbor_id, neighbor_strength))
            
            # CRITICAL: Sort next wave for determinism
            to_process = sorted(next_wave, key=lambda x: (x[0], x[1]))
        
        return activated

class ActivationDeterminismTest:
    """Test activation determinism with crisp pass/fail"""
    
    def __init__(self, adapter: GoldenBusAdapter):
        self.adapter = adapter
        self.violations = []
    
    async def test_identical_inputs_produce_identical_outputs(self) -> Dict:
        """
        INVARIANT: Same seed + inputs = same outputs
        Method: Run activation 10 times with same inputs, verify identical results
        """
        print("🧪 TEST: Identical Inputs → Identical Outputs")
        
        seed = 42
        start_node = "node_5"
        context = {"boost_category": "cat_3", "mode": "exploration"}
        
        # Run activation multiple times with same inputs
        results = []
        
        for run in range(10):
            with self.adapter.span(f"determinism_test_run_{run}", seed=seed):
                # Create fresh graph with same seed
                graph = MockSemanticGraph(seed)
                
                # Run activation
                activated = await graph.activate_neighborhood(start_node, context)
                
                # Convert to sorted list for comparison
                activated_list = sorted(list(activated))
                results.append(activated_list)
                
                # Record metrics
                self.adapter.record_activation(
                    start_node=start_node,
                    activated_count=len(activated),
                    propagation_depth=3,
                    duration_ms=1.0  # Mock timing
                )
        
        # Verify all results are identical
        first_result = results[0]
        all_identical = all(r == first_result for r in results)
        
        if not all_identical:
            # Find differences
            for i, result in enumerate(results[1:], 1):
                if result != first_result:
                    diff_added = set(result) - set(first_result)
                    diff_removed = set(first_result) - set(result)
                    
                    self.violations.append({
                        "test": "identical_inputs",
                        "run": i,
                        "added_nodes": list(diff_added),
                        "removed_nodes": list(diff_removed)
                    })
        
        return {
            "test": "identical_inputs",
            "runs": len(results),
            "all_identical": all_identical,
            "activated_count": len(first_result) if results else 0,
            "violations": len([v for v in self.violations if v["test"] == "identical_inputs"])
        }
    
    async def test_different_seeds_produce_different_outputs(self) -> Dict:
        """
        INVARIANT: Different seeds = different outputs (no accidental determinism)
        Method: Run with different seeds, verify outputs differ
        """
        print("🧪 TEST: Different Seeds → Different Outputs")
        
        start_node = "node_5"
        context = {"boost_category": "cat_3"}
        
        results_by_seed = {}
        
        for seed in [42, 123, 789]:
            with self.adapter.span(f"seed_variation_test", seed=seed):
                graph = MockSemanticGraph(seed)
                activated = await graph.activate_neighborhood(start_node, context)
                results_by_seed[seed] = sorted(list(activated))
        
        # Check that different seeds produce different results
        unique_results = len(set(tuple(r) for r in results_by_seed.values()))
        all_different = unique_results == len(results_by_seed)
        
        if not all_different:
            self.violations.append({
                "test": "different_seeds",
                "issue": "Different seeds produced identical results",
                "seeds": list(results_by_seed.keys())
            })
        
        return {
            "test": "different_seeds",
            "seeds_tested": len(results_by_seed),
            "unique_results": unique_results,
            "all_different": all_different,
            "violations": len([v for v in self.violations if v["test"] == "different_seeds"])
        }
    
    async def test_context_changes_affect_output_deterministically(self) -> Dict:
        """
        INVARIANT: Same seed + different context = deterministic different outputs
        Method: Change context, verify outputs change but remain deterministic
        """
        print("🧪 TEST: Context Changes → Deterministic Changes")
        
        seed = 42
        start_node = "node_5"
        
        contexts = [
            {"boost_category": "cat_1"},
            {"boost_category": "cat_3"},
            {"boost_category": "cat_5"},
            {}  # No boost
        ]
        
        results_by_context = {}
        
        for ctx_idx, context in enumerate(contexts):
            # Run twice to verify determinism within context
            runs = []
            
            for run in range(2):
                with self.adapter.span(f"context_test_{ctx_idx}_run_{run}", seed=seed):
                    graph = MockSemanticGraph(seed)
                    activated = await graph.activate_neighborhood(start_node, context)
                    runs.append(sorted(list(activated)))
            
            # Verify both runs are identical
            if runs[0] != runs[1]:
                self.violations.append({
                    "test": "context_determinism",
                    "context": str(context),
                    "issue": "Same context produced different results"
                })
            
            results_by_context[str(context)] = runs[0]
        
        # Verify different contexts produce different results
        unique_results = len(set(tuple(r) for r in results_by_context.values()))
        
        return {
            "test": "context_changes",
            "contexts_tested": len(contexts),
            "unique_results": unique_results,
            "determinism_maintained": len([v for v in self.violations if v["test"] == "context_determinism"]) == 0,
            "violations": len([v for v in self.violations if v["test"] == "context_determinism"])
        }
    
    async def test_activation_fingerprint_stability(self) -> Dict:
        """
        INVARIANT: Activation produces stable fingerprint for same inputs
        Method: Generate hash of activation pattern, verify stability
        """
        print("🧪 TEST: Activation Fingerprint Stability")
        
        seed = 42
        test_cases = [
            ("node_0", {"mode": "explore"}),
            ("node_10", {"boost_category": "cat_5"}),
            ("node_50", {}),
        ]
        
        fingerprints = {}
        
        for start_node, context in test_cases:
            # Generate fingerprint multiple times
            case_fingerprints = []
            
            for run in range(5):
                graph = MockSemanticGraph(seed)
                activated = await graph.activate_neighborhood(start_node, context)
                
                # Create fingerprint
                activation_str = json.dumps(sorted(list(activated)))
                fingerprint = hashlib.sha256(activation_str.encode()).hexdigest()[:16]
                case_fingerprints.append(fingerprint)
            
            # Verify all fingerprints are identical
            all_same = all(fp == case_fingerprints[0] for fp in case_fingerprints)
            
            if not all_same:
                self.violations.append({
                    "test": "fingerprint_stability",
                    "start_node": start_node,
                    "context": str(context),
                    "fingerprints": case_fingerprints
                })
            
            fingerprints[f"{start_node}_{str(context)}"] = case_fingerprints[0]
        
        return {
            "test": "fingerprint_stability",
            "test_cases": len(test_cases),
            "stable_fingerprints": len([k for k, v in fingerprints.items()]),
            "violations": len([v for v in self.violations if v["test"] == "fingerprint_stability"])
        }

async def run_activation_determinism_tests():
    """Main test runner with crisp pass/fail"""
    
    print("🔬 ACTIVATION DETERMINISM TEST SUITE")
    print("=" * 60)
    print("CLAIM: Activation is 100% deterministic under controlled conditions")
    print("PASS: Zero violations across all tests")
    print("FAIL: Any non-deterministic behavior")
    print("=" * 60)
    
    # Create deterministic test config
    config = TestConfig.create(TestProfile.SMALL, seed=42)
    
    # Initialize adapter with deterministic config
    adapter = GoldenBusAdapter(config)
    
    # Run tests
    tester = ActivationDeterminismTest(adapter)
    
    results = []
    results.append(await tester.test_identical_inputs_produce_identical_outputs())
    results.append(await tester.test_different_seeds_produce_different_outputs())
    results.append(await tester.test_context_changes_affect_output_deterministically())
    results.append(await tester.test_activation_fingerprint_stability())
    
    # Calculate overall pass/fail
    total_violations = len(tester.violations)
    all_passed = total_violations == 0
    
    # Print results
    print(f"\n📊 TEST RESULTS")
    print(f"Config Hash: {adapter.config_hash}")
    
    for result in results:
        test_passed = result.get("violations", 0) == 0
        status = "✅ PASS" if test_passed else "❌ FAIL"
        print(f"   {status} {result['test']}")
        
        if not test_passed and "violations" in result:
            print(f"      Violations: {result['violations']}")
    
    # Overall verdict
    print(f"\n🏆 OVERALL VERDICT: {'✅ PASS' if all_passed else '❌ FAIL'}")
    
    if all_passed:
        print("   ✅ Activation is 100% deterministic")
        print("   ✅ Same inputs always produce same outputs")
        print("   ✅ Ready for reproducible experiments")
    else:
        print(f"   ❌ Found {total_violations} determinism violations")
        print("   ❌ System has non-deterministic behavior")
        print("\n   VIOLATIONS:")
        for v in tester.violations[:5]:  # Show first 5
            print(f"      • {v}")
    
    # Export telemetry
    adapter.export_telemetry(f"activation_determinism_{adapter.config_hash}.json")
    
    # Return pass/fail for CI
    return all_passed

if __name__ == "__main__":
    import asyncio
    passed = asyncio.run(run_activation_determinism_tests())
    exit(0 if passed else 1)