"""
Chaos Validation Suite - Multiple runs with variance analysis
Ensures the GAIA architecture's resilience is consistent, not a fluke
"""

import asyncio
import time
import json
import statistics
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.stress.chaos_convergence import ChaosConvergenceTest, ChaosLevel

@dataclass
class ChaosRunResult:
    """Results from a single chaos run"""
    run_id: int
    chaos_level: str
    duration: float
    system_survived: bool
    peak_memory_mb: float
    peak_cpu_percent: float
    total_thoughts: int
    success_rate: float
    zones_created: int
    fragmentation_percent: float
    semantic_collisions: int
    cascade_depth: int
    error_messages: List[str]

class ChaosValidationSuite:
    """Runs multiple chaos tests and analyzes variance"""
    
    def __init__(self):
        self.results: List[ChaosRunResult] = []
        self.variance_analysis: Dict[str, Any] = {}
    
    async def run_validation_suite(self, num_runs: int = 5, base_duration: int = 30):
        """Run multiple chaos tests with different parameters"""
        print(f"🔬 CHAOS VALIDATION SUITE")
        print(f"Running {num_runs} tests to validate consistency")
        print("=" * 60)
        
        test_configurations = [
            (ChaosLevel.MODERATE_CHAOS, base_duration),
            (ChaosLevel.SEVERE_CHAOS, base_duration),
            (ChaosLevel.SEVERE_CHAOS, base_duration * 2),  # Longer duration
            (ChaosLevel.APOCALYPTIC_CHAOS, base_duration),
            (ChaosLevel.APOCALYPTIC_CHAOS, base_duration // 2),  # Shorter but intense
        ]
        
        for run_id in range(num_runs):
            chaos_level, duration = test_configurations[run_id % len(test_configurations)]
            
            print(f"\n🧪 RUN {run_id + 1}/{num_runs}")
            print(f"Level: {chaos_level.value}, Duration: {duration}s")
            print("-" * 40)
            
            try:
                result = await self._run_single_chaos_test(run_id, chaos_level, duration)
                self.results.append(result)
                
                # Brief summary
                survival_status = "✅ SURVIVED" if result.system_survived else "💀 CRASHED"
                print(f"Result: {survival_status} - {result.success_rate:.1%} success rate")
                
                # Brief cooldown between runs
                if run_id < num_runs - 1:
                    print("😴 Cooling down for 3 seconds...")
                    await asyncio.sleep(3)
                    
            except Exception as e:
                print(f"💥 Run {run_id + 1} failed with exception: {e}")
                # Create failure result
                failure_result = ChaosRunResult(
                    run_id=run_id,
                    chaos_level=chaos_level.value,
                    duration=0,
                    system_survived=False,
                    peak_memory_mb=0,
                    peak_cpu_percent=0,
                    total_thoughts=0,
                    success_rate=0.0,
                    zones_created=0,
                    fragmentation_percent=0,
                    semantic_collisions=0,
                    cascade_depth=0,
                    error_messages=[str(e)]
                )
                self.results.append(failure_result)
        
        # Analyze variance
        self._analyze_variance()
        
        # Generate comprehensive report
        self._print_validation_report()
        
        return self.results
    
    async def _run_single_chaos_test(self, run_id: int, chaos_level: ChaosLevel, 
                                    duration: int) -> ChaosRunResult:
        """Run a single chaos test and extract results"""
        chaos_test = ChaosConvergenceTest(chaos_level)
        error_messages = []
        
        try:
            await chaos_test.initialize_chaos_environment()
            
            start_time = time.time()
            await chaos_test.execute_chaos_convergence(duration)
            actual_duration = time.time() - start_time
            
            # Get detailed report
            report = chaos_test.generate_chaos_report()
            
            return ChaosRunResult(
                run_id=run_id,
                chaos_level=chaos_level.value,
                duration=actual_duration,
                system_survived=report['system_survived'],
                peak_memory_mb=report['metrics']['peak_memory_mb'],
                peak_cpu_percent=report['metrics']['peak_cpu_percent'],
                total_thoughts=report['metrics']['total_thoughts_spawned'],
                success_rate=report['metrics']['success_rate'],
                zones_created=report['metrics']['zones_created'],
                fragmentation_percent=report['metrics']['btree_fragmentation_percent'],
                semantic_collisions=report['attack_analysis']['memory_fragmentation']['semantic_collisions'],
                cascade_depth=report['attack_analysis']['failure_cascade']['cascade_depth'],
                error_messages=error_messages
            )
            
        except Exception as e:
            error_messages.append(str(e))
            raise
        finally:
            await chaos_test.cleanup()
    
    def _analyze_variance(self):
        """Analyze variance across all runs"""
        if len(self.results) < 2:
            return
        
        # Filter successful runs for stats
        successful_runs = [r for r in self.results if r.system_survived]
        
        if not successful_runs:
            self.variance_analysis = {"error": "No successful runs to analyze"}
            return
        
        # Calculate statistics for key metrics
        metrics = {
            'peak_memory_mb': [r.peak_memory_mb for r in successful_runs],
            'total_thoughts': [r.total_thoughts for r in successful_runs],
            'success_rate': [r.success_rate for r in successful_runs],
            'zones_created': [r.zones_created for r in successful_runs],
            'fragmentation_percent': [r.fragmentation_percent for r in successful_runs],
            'duration': [r.duration for r in successful_runs]
        }
        
        self.variance_analysis = {}
        
        for metric_name, values in metrics.items():
            if len(values) > 1:
                self.variance_analysis[metric_name] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values),
                    'coefficient_of_variation': (statistics.stdev(values) / statistics.mean(values)) * 100 if statistics.mean(values) != 0 else 0,
                    'values': values
                }
        
        # Calculate survival rate
        total_runs = len(self.results)
        survived_runs = len(successful_runs)
        self.variance_analysis['survival_rate'] = {
            'percentage': (survived_runs / total_runs) * 100,
            'survived': survived_runs,
            'total': total_runs
        }
        
        # Identify outliers (values > 2 standard deviations from mean)
        self.variance_analysis['outliers'] = {}
        for metric_name, stats in self.variance_analysis.items():
            if isinstance(stats, dict) and 'stdev' in stats and 'mean' in stats:
                threshold = 2 * stats['stdev']
                mean = stats['mean']
                outliers = []
                
                for i, value in enumerate(stats['values']):
                    if abs(value - mean) > threshold:
                        outliers.append({
                            'run_id': successful_runs[i].run_id,
                            'value': value,
                            'deviation': abs(value - mean)
                        })
                
                if outliers:
                    self.variance_analysis['outliers'][metric_name] = outliers
    
    def _print_validation_report(self):
        """Print comprehensive validation report"""
        print(f"\n" + "=" * 80)
        print(f"🔬 CHAOS VALIDATION RESULTS")
        print(f"=" * 80)
        
        total_runs = len(self.results)
        successful_runs = [r for r in self.results if r.system_survived]
        failed_runs = [r for r in self.results if not r.system_survived]
        
        # Overall summary
        print(f"\n📊 OVERALL SUMMARY:")
        print(f"  Total Runs: {total_runs}")
        print(f"  Successful: {len(successful_runs)} ({len(successful_runs)/total_runs:.1%})")
        print(f"  Failed: {len(failed_runs)} ({len(failed_runs)/total_runs:.1%})")
        
        # Per-run details
        print(f"\n📋 RUN DETAILS:")
        for result in self.results:
            status = "✅ PASS" if result.system_survived else "💀 FAIL"
            print(f"  Run {result.run_id + 1}: {status} | Level: {result.chaos_level} | "
                  f"Thoughts: {result.total_thoughts} | Success: {result.success_rate:.1%} | "
                  f"Memory: {result.peak_memory_mb:.1f}MB")
        
        # Variance analysis
        if 'survival_rate' in self.variance_analysis:
            print(f"\n📈 CONSISTENCY ANALYSIS:")
            survival = self.variance_analysis['survival_rate']
            print(f"  System Survival Rate: {survival['percentage']:.1f}% ({survival['survived']}/{survival['total']})")
            
            if len(successful_runs) > 1:
                print(f"\n📊 PERFORMANCE VARIANCE:")
                
                key_metrics = ['peak_memory_mb', 'total_thoughts', 'success_rate', 'zones_created']
                for metric in key_metrics:
                    if metric in self.variance_analysis:
                        stats = self.variance_analysis[metric]
                        cv = stats['coefficient_of_variation']
                        
                        variance_level = "LOW" if cv < 10 else "MODERATE" if cv < 25 else "HIGH"
                        print(f"  {metric.replace('_', ' ').title()}:")
                        print(f"    Mean: {stats['mean']:.2f} ± {stats['stdev']:.2f}")
                        print(f"    Range: {stats['min']:.2f} - {stats['max']:.2f}")
                        print(f"    Variance: {variance_level} (CV: {cv:.1f}%)")
        
        # Outlier analysis
        if 'outliers' in self.variance_analysis and self.variance_analysis['outliers']:
            print(f"\n⚠️ OUTLIERS DETECTED:")
            for metric, outliers in self.variance_analysis['outliers'].items():
                print(f"  {metric.replace('_', ' ').title()}:")
                for outlier in outliers:
                    print(f"    Run {outlier['run_id'] + 1}: {outlier['value']:.2f} "
                          f"(deviation: {outlier['deviation']:.2f})")
        
        # Failure analysis
        if failed_runs:
            print(f"\n💀 FAILURE ANALYSIS:")
            failure_patterns = {}
            for failure in failed_runs:
                level = failure.chaos_level
                if level not in failure_patterns:
                    failure_patterns[level] = []
                failure_patterns[level].append(failure)
            
            for level, failures in failure_patterns.items():
                print(f"  {level}: {len(failures)} failures")
                for failure in failures:
                    if failure.error_messages:
                        print(f"    Run {failure.run_id + 1}: {failure.error_messages[0][:100]}...")
        
        # Final verdict
        print(f"\n🏆 VALIDATION VERDICT:")
        
        if len(successful_runs) == total_runs:
            print("🟢 EXCELLENT CONSISTENCY - All runs successful")
            if self.variance_analysis:
                avg_cv = statistics.mean([
                    stats.get('coefficient_of_variation', 0) 
                    for stats in self.variance_analysis.values() 
                    if isinstance(stats, dict) and 'coefficient_of_variation' in stats
                ])
                if avg_cv < 15:
                    print("   Performance variance is LOW - System behavior is highly predictable")
                elif avg_cv < 30:
                    print("   Performance variance is MODERATE - System behavior is reasonably consistent")
                else:
                    print("   Performance variance is HIGH - System behavior varies significantly")
        
        elif len(successful_runs) >= total_runs * 0.8:
            print("🟡 GOOD CONSISTENCY - Most runs successful")
            print("   Some instability under extreme conditions, but generally reliable")
        
        elif len(successful_runs) >= total_runs * 0.5:
            print("🟠 MODERATE CONSISTENCY - Mixed results")
            print("   System shows resilience but has reliability issues under stress")
        
        else:
            print("🔴 POOR CONSISTENCY - Frequent failures")
            print("   System is not ready for production - significant stability issues")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if len(failed_runs) == 0:
            print("   ✅ System is production-ready with excellent resilience")
            print("   ✅ Consider deploying with confidence in high-stress environments")
        elif len(failed_runs) < total_runs * 0.2:
            print("   ⚡ System is mostly reliable - monitor edge cases in production")
            print("   ⚡ Consider adding additional safeguards for extreme loads")
        else:
            print("   🔧 System needs improvement before production deployment")
            print("   🔧 Focus on failure modes and recovery mechanisms")

async def run_chaos_validation():
    """Run the complete chaos validation suite"""
    validator = ChaosValidationSuite()
    
    print("🌪️ GAIA CHAOS VALIDATION SUITE")
    print("Testing system consistency across multiple chaos scenarios")
    print("This will take several minutes...")
    
    results = await validator.run_validation_suite(num_runs=5, base_duration=20)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"chaos_validation_{timestamp}.json"
    
    validation_data = {
        "timestamp": timestamp,
        "total_runs": len(results),
        "results": [asdict(r) for r in results],
        "variance_analysis": validator.variance_analysis
    }
    
    with open(filename, 'w') as f:
        json.dump(validation_data, f, indent=2)
    
    print(f"\n💾 Detailed results saved to {filename}")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_chaos_validation())