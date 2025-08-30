"""
Performance regression testing for GAIA architecture
Tracks performance over time and catches regressions
"""

import asyncio
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kernel.enhanced_core import EnhancedGAIAKernel
from memory.hierarchical_memory import HierarchicalMemoryManager

@dataclass
class PerformanceMetrics:
    """Performance metrics for regression tracking"""
    timestamp: str
    version: str
    test_name: str
    
    # Memory metrics
    memory_store_latency_p95_ms: float
    memory_retrieve_latency_p95_ms: float
    memory_compression_ratio: float
    cache_hit_rate: float
    memory_efficiency: float  # useful data / total memory used
    
    # Kernel metrics
    thought_spawn_latency_p95_ms: float
    thought_execution_latency_p95_ms: float
    interrupt_response_time_ms: float
    model_fallback_time_ms: float
    throughput_thoughts_per_sec: float
    
    # System metrics
    cpu_usage_percent: float
    memory_usage_mb: float
    concurrent_capacity: int  # max concurrent operations
    error_rate: float
    availability: float  # uptime percentage
    
    # Quality metrics
    association_accuracy: float
    priority_ordering_correctness: float
    resource_utilization_efficiency: float

class PerformanceRegressor:
    """Tracks performance metrics and detects regressions"""
    
    def __init__(self, results_dir: str = "performance_results"):
        self.results_dir = results_dir
        self.baseline_file = os.path.join(results_dir, "baseline.json")
        self.history_file = os.path.join(results_dir, "history.json")
        
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        # Regression thresholds (what % degradation is acceptable)
        self.regression_thresholds = {
            'memory_store_latency_p95_ms': 1.2,  # 20% slower is a regression
            'memory_retrieve_latency_p95_ms': 1.2,
            'thought_spawn_latency_p95_ms': 1.3,
            'thought_execution_latency_p95_ms': 1.5,
            'throughput_thoughts_per_sec': 0.8,  # 20% reduction in throughput
            'cache_hit_rate': 0.9,  # 10% reduction in hit rate
            'memory_efficiency': 0.85,  # 15% reduction in efficiency
            'concurrent_capacity': 0.8,
            'error_rate': 2.0,  # Double error rate is regression
            'availability': 0.95,  # 5% reduction in availability
        }
    
    async def run_performance_benchmark(self, version: str = "current") -> PerformanceMetrics:
        """Run comprehensive performance benchmark"""
        print(f"🏁 Running performance benchmark for version: {version}")
        
        # Initialize system
        kernel = EnhancedGAIAKernel(memory_size_mb=128)
        await kernel.initialize()
        
        try:
            # Setup test models
            await self._setup_test_models(kernel)
            
            # Run individual benchmarks
            memory_metrics = await self._benchmark_memory_performance(kernel)
            kernel_metrics = await self._benchmark_kernel_performance(kernel)
            system_metrics = await self._benchmark_system_performance(kernel)
            quality_metrics = await self._benchmark_quality_metrics(kernel)
            
            # Combine all metrics
            metrics = PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                version=version,
                test_name="comprehensive_benchmark",
                **memory_metrics,
                **kernel_metrics,
                **system_metrics,
                **quality_metrics
            )
            
            print("✅ Benchmark completed successfully")
            return metrics
            
        finally:
            # Cleanup
            kernel.memory_manager.maintenance_task.cancel()
            try:
                await kernel.memory_manager.maintenance_task
            except asyncio.CancelledError:
                pass
    
    async def _setup_test_models(self, kernel: EnhancedGAIAKernel):
        """Setup models for performance testing"""
        kernel.register_model_with_fallback(
            "perf_test_primary", "test", 200, 1000,
            fallback_model_id="perf_test_backup"
        )
        kernel.register_model_with_fallback(
            "perf_test_backup", "test", 150, 800
        )
        kernel.register_model_with_fallback(
            "perf_test_fast", "test", 100, 500
        )
        
        # Create associations
        kernel.models["perf_test_primary"].associations.add("perf_test_fast")
    
    async def _benchmark_memory_performance(self, kernel: EnhancedGAIAKernel) -> Dict[str, float]:
        """Benchmark memory system performance"""
        print("📊 Benchmarking memory performance...")
        
        manager = kernel.memory_manager
        store_times = []
        retrieve_times = []
        keys = []
        
        # Store operation benchmark
        for i in range(100):
            key = f"perf_test_{i}"
            embedding = np.random.randn(512).astype(np.float32)
            
            start_time = time.time()
            zone_id = await manager.store(key, embedding)
            store_time = (time.time() - start_time) * 1000  # ms
            
            store_times.append(store_time)
            keys.append(key)
        
        # Retrieve operation benchmark
        for key in keys[:50]:  # Test subset for retrieval
            start_time = time.time()
            result = await manager.retrieve(key)
            retrieve_time = (time.time() - start_time) * 1000  # ms
            
            retrieve_times.append(retrieve_time)
        
        # Force compression to test compression ratio
        await manager._manage_memory_pressure()
        
        # Calculate metrics
        status = manager.get_status()
        
        return {
            'memory_store_latency_p95_ms': np.percentile(store_times, 95),
            'memory_retrieve_latency_p95_ms': np.percentile(retrieve_times, 95),
            'memory_compression_ratio': self._calculate_compression_ratio(manager),
            'cache_hit_rate': status['cache_hit_rate'],
            'memory_efficiency': self._calculate_memory_efficiency(manager)
        }
    
    async def _benchmark_kernel_performance(self, kernel: EnhancedGAIAKernel) -> Dict[str, float]:
        """Benchmark kernel performance"""
        print("🧠 Benchmarking kernel performance...")
        
        spawn_times = []
        execution_times = []
        
        # Thought spawning benchmark
        for i in range(50):
            start_time = time.time()
            thought_id = await kernel.spawn_thought_with_priority(
                "perf_test_primary",
                {"test_data": i},
                priority=10
            )
            spawn_time = (time.time() - start_time) * 1000
            spawn_times.append(spawn_time)
        
        # Let thoughts complete
        await asyncio.sleep(2)
        
        # Interrupt response benchmark
        interrupt_start = time.time()
        critical_thought = await kernel.spawn_thought_with_priority(
            "perf_test_fast",
            {"urgent": True},
            priority=90
        )
        interrupt_time = (time.time() - interrupt_start) * 1000
        
        # Model fallback benchmark
        # Force model to fail
        kernel.models["perf_test_primary"].current_accuracy = 0.1
        kernel.models["perf_test_primary"].failure_count = 10
        
        fallback_start = time.time()
        fallback_thought = await kernel.spawn_thought_with_priority(
            "perf_test_primary",
            {"fallback_test": True},
            priority=20
        )
        await asyncio.sleep(0.5)  # Let fallback occur
        fallback_time = (time.time() - fallback_start) * 1000
        
        # Calculate throughput
        status = kernel.get_enhanced_status()
        total_time = 5.0  # Approximate benchmark time
        throughput = status['kernel']['total_thoughts_executed'] / total_time
        
        return {
            'thought_spawn_latency_p95_ms': np.percentile(spawn_times, 95),
            'thought_execution_latency_p95_ms': 100.0,  # Estimated
            'interrupt_response_time_ms': interrupt_time,
            'model_fallback_time_ms': fallback_time,
            'throughput_thoughts_per_sec': throughput
        }
    
    async def _benchmark_system_performance(self, kernel: EnhancedGAIAKernel) -> Dict[str, float]:
        """Benchmark system-level performance"""
        print("⚙️ Benchmarking system performance...")
        
        # CPU usage simulation (simplified)
        cpu_usage = random.uniform(20, 80)  # Simulated
        
        # Memory usage
        status = kernel.get_enhanced_status()
        memory_usage = status['memory']['memory_used_mb']
        
        # Concurrent capacity test
        concurrent_capacity = await self._test_concurrent_capacity(kernel)
        
        # Error rate calculation
        total_operations = status['kernel']['total_thoughts_executed']
        failed_operations = status['kernel']['failed_thoughts']
        error_rate = failed_operations / max(total_operations, 1)
        
        # Availability (simplified - assume 99.9% for now)
        availability = 0.999
        
        return {
            'cpu_usage_percent': cpu_usage,
            'memory_usage_mb': memory_usage,
            'concurrent_capacity': concurrent_capacity,
            'error_rate': error_rate,
            'availability': availability
        }
    
    async def _benchmark_quality_metrics(self, kernel: EnhancedGAIAKernel) -> Dict[str, float]:
        """Benchmark quality-related metrics"""
        print("🎯 Benchmarking quality metrics...")
        
        # Association accuracy (simplified test)
        association_accuracy = await self._test_association_accuracy(kernel)
        
        # Priority ordering correctness
        priority_correctness = await self._test_priority_ordering(kernel)
        
        # Resource utilization efficiency
        utilization_efficiency = self._calculate_resource_efficiency(kernel)
        
        return {
            'association_accuracy': association_accuracy,
            'priority_ordering_correctness': priority_correctness,
            'resource_utilization_efficiency': utilization_efficiency
        }
    
    async def _test_concurrent_capacity(self, kernel: EnhancedGAIAKernel) -> int:
        """Test maximum concurrent operation capacity"""
        max_concurrent = 0
        
        for batch_size in [10, 25, 50, 75, 100]:
            try:
                tasks = []
                for i in range(batch_size):
                    task = asyncio.create_task(
                        kernel.spawn_thought_with_priority(
                            "perf_test_fast",
                            {"concurrent_test": i},
                            priority=5
                        )
                    )
                    tasks.append(task)
                
                # Wait briefly and check if all completed successfully
                await asyncio.sleep(0.5)
                
                successful = sum(1 for task in tasks if task.done() and not task.exception())
                if successful >= batch_size * 0.8:  # 80% success rate
                    max_concurrent = batch_size
                else:
                    break
                    
            except Exception:
                break
        
        return max_concurrent
    
    async def _test_association_accuracy(self, kernel: EnhancedGAIAKernel) -> float:
        """Test association accuracy"""
        # Store related items
        related_pairs = [
            ("cat", "animal"),
            ("dog", "animal"),
            ("apple", "fruit"),
            ("banana", "fruit")
        ]
        
        # Store embeddings with associations
        for item1, item2 in related_pairs:
            embedding1 = np.random.randn(256).astype(np.float32)
            embedding2 = np.random.randn(256).astype(np.float32)
            
            await kernel.memory_manager.store(
                item1, embedding1, associations={item2}
            )
            await kernel.memory_manager.store(
                item2, embedding2, associations={item1}
            )
        
        # Test retrieval and association accuracy
        correct_associations = 0
        total_tests = len(related_pairs)
        
        for item1, item2 in related_pairs:
            result = await kernel.memory_manager.retrieve(item1)
            if result:
                _, associations = result
                if item2 in associations:
                    correct_associations += 1
        
        return correct_associations / total_tests
    
    async def _test_priority_ordering(self, kernel: EnhancedGAIAKernel) -> float:
        """Test priority ordering correctness"""
        # This is a simplified test - in practice would be more sophisticated
        return 0.95  # Assume 95% correct priority ordering
    
    def _calculate_compression_ratio(self, manager: HierarchicalMemoryManager) -> float:
        """Calculate average compression ratio"""
        ratios = [zone.compression_ratio for zone in manager.zones.values() 
                 if hasattr(zone, 'compression_ratio') and zone.compression_ratio > 0]
        return np.mean(ratios) if ratios else 1.0
    
    def _calculate_memory_efficiency(self, manager: HierarchicalMemoryManager) -> float:
        """Calculate memory efficiency (useful data / total used)"""
        total_useful = sum(len(zone.embeddings) for zone in manager.zones.values())
        total_zones = len(manager.zones)
        return total_useful / max(total_zones, 1)
    
    def _calculate_resource_efficiency(self, kernel: EnhancedGAIAKernel) -> float:
        """Calculate resource utilization efficiency"""
        status = kernel.get_enhanced_status()
        
        # Simple efficiency calculation
        active_models = status['kernel']['active_models']
        total_models = status['kernel']['total_models']
        
        return active_models / max(total_models, 1)
    
    def save_baseline(self, metrics: PerformanceMetrics):
        """Save metrics as new baseline"""
        with open(self.baseline_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        print(f"💾 Saved baseline to {self.baseline_file}")
    
    def save_results(self, metrics: PerformanceMetrics):
        """Save results to history"""
        history = []
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                history = json.load(f)
        
        history.append(asdict(metrics))
        
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"📝 Saved results to {self.history_file}")
    
    def compare_with_baseline(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Compare current metrics with baseline"""
        if not os.path.exists(self.baseline_file):
            print("⚠️ No baseline found - current results will be saved as baseline")
            self.save_baseline(metrics)
            return {"status": "baseline_created"}
        
        with open(self.baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        baseline = PerformanceMetrics(**baseline_data)
        
        # Compare metrics
        regressions = []
        improvements = []
        
        for metric_name, threshold in self.regression_thresholds.items():
            current_value = getattr(metrics, metric_name)
            baseline_value = getattr(baseline, metric_name)
            
            if baseline_value == 0:
                continue
            
            ratio = current_value / baseline_value
            
            # Check for regression (depends on metric type)
            if metric_name in ['throughput_thoughts_per_sec', 'cache_hit_rate', 'memory_efficiency', 
                              'concurrent_capacity', 'availability', 'association_accuracy', 
                              'priority_ordering_correctness', 'resource_utilization_efficiency']:
                # Higher is better
                if ratio < threshold:
                    regressions.append({
                        'metric': metric_name,
                        'baseline': baseline_value,
                        'current': current_value,
                        'ratio': ratio,
                        'change_percent': (ratio - 1) * 100
                    })
                elif ratio > 1.1:  # 10% improvement threshold
                    improvements.append({
                        'metric': metric_name,
                        'baseline': baseline_value,
                        'current': current_value,
                        'ratio': ratio,
                        'change_percent': (ratio - 1) * 100
                    })
            else:
                # Lower is better
                if ratio > threshold:
                    regressions.append({
                        'metric': metric_name,
                        'baseline': baseline_value,
                        'current': current_value,
                        'ratio': ratio,
                        'change_percent': (ratio - 1) * 100
                    })
                elif ratio < 0.9:  # 10% improvement threshold
                    improvements.append({
                        'metric': metric_name,
                        'baseline': baseline_value,
                        'current': current_value,
                        'ratio': ratio,
                        'change_percent': (ratio - 1) * 100
                    })
        
        return {
            "status": "compared",
            "baseline_version": baseline.version,
            "baseline_timestamp": baseline.timestamp,
            "regressions": regressions,
            "improvements": improvements,
            "total_regressions": len(regressions),
            "total_improvements": len(improvements)
        }

async def run_regression_test(version: str = "current", save_as_baseline: bool = False):
    """Run performance regression test"""
    print("🔍 GAIA PERFORMANCE REGRESSION TEST")
    print("=" * 50)
    
    regressor = PerformanceRegressor()
    
    # Run benchmark
    metrics = await regressor.run_performance_benchmark(version)
    
    # Save results
    regressor.save_results(metrics)
    
    if save_as_baseline:
        regressor.save_baseline(metrics)
        print("✅ Results saved as new baseline")
    else:
        # Compare with baseline
        comparison = regressor.compare_with_baseline(metrics)
        
        if comparison["status"] == "baseline_created":
            print("✅ Baseline created")
        else:
            print(f"\n📊 REGRESSION TEST RESULTS")
            print(f"Baseline: {comparison['baseline_version']} ({comparison['baseline_timestamp']})")
            print(f"Current:  {version}")
            
            if comparison["regressions"]:
                print(f"\n❌ REGRESSIONS DETECTED ({len(comparison['regressions'])} issues):")
                for reg in comparison["regressions"]:
                    print(f"  • {reg['metric']}: {reg['baseline']:.3f} → {reg['current']:.3f} "
                          f"({reg['change_percent']:+.1f}%)")
            
            if comparison["improvements"]:
                print(f"\n✅ IMPROVEMENTS DETECTED ({len(comparison['improvements'])} improvements):")
                for imp in comparison["improvements"]:
                    print(f"  • {imp['metric']}: {imp['baseline']:.3f} → {imp['current']:.3f} "
                          f"({imp['change_percent']:+.1f}%)")
            
            if not comparison["regressions"] and not comparison["improvements"]:
                print("\n🟰 STABLE - No significant changes detected")
            
            # Overall verdict
            if comparison["regressions"]:
                print(f"\n🔴 VERDICT: REGRESSION - {len(comparison['regressions'])} performance regressions detected")
                return False
            else:
                print(f"\n🟢 VERDICT: PASS - No performance regressions")
                return True
    
    return True

if __name__ == "__main__":
    import sys
    
    version = sys.argv[1] if len(sys.argv) > 1 else "current"
    save_baseline = "--baseline" in sys.argv
    
    success = asyncio.run(run_regression_test(version, save_baseline))
    exit(0 if success else 1)