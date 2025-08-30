import asyncio
import sys
import os
import statistics
sys.path.append('tests/stress')

from chaos_convergence import ChaosConvergenceTest, ChaosLevel

async def quick_validation():
    print('🔬 QUICK CHAOS VALIDATION - 3 runs')
    results = []
    
    test_configs = [
        (ChaosLevel.MODERATE_CHAOS, 15),
        (ChaosLevel.SEVERE_CHAOS, 15), 
        (ChaosLevel.APOCALYPTIC_CHAOS, 10)
    ]
    
    for i, (level, duration) in enumerate(test_configs):
        print(f'\n🧪 RUN {i+1}/3: {level.value} for {duration}s')
        
        chaos_test = ChaosConvergenceTest(level)
        try:
            await chaos_test.initialize_chaos_environment()
            await chaos_test.execute_chaos_convergence(duration)
            
            report = chaos_test.generate_chaos_report()
            results.append({
                'run': i+1,
                'level': level.value,
                'survived': report['system_survived'],
                'thoughts': report['metrics']['total_thoughts_spawned'],
                'success_rate': report['metrics']['success_rate'],
                'memory_mb': report['metrics']['peak_memory_mb'],
                'zones': report['metrics']['zones_created'],
                'fragmentation': report['metrics']['btree_fragmentation_percent']
            })
            
            status = '✅ SURVIVED' if report['system_survived'] else '💀 CRASHED'
            print(f'   Result: {status} - {report["metrics"]["success_rate"]:.1%} success, {report["metrics"]["total_thoughts_spawned"]} thoughts')
            
        except Exception as e:
            print(f'   💥 Exception: {str(e)[:100]}...')
            results.append({'run': i+1, 'level': level.value, 'survived': False, 'error': str(e)})
        finally:
            await chaos_test.cleanup()
    
    # Analysis
    print(f'\n📊 VALIDATION SUMMARY:')
    survived = sum(1 for r in results if r.get('survived', False))
    print(f'   Survival Rate: {survived}/{len(results)} ({survived/len(results):.1%})')
    
    if survived > 0:
        successful_results = [r for r in results if r.get('survived', False)]
        
        avg_thoughts = sum(r.get('thoughts', 0) for r in successful_results) / len(successful_results)
        avg_success = sum(r.get('success_rate', 0) for r in successful_results) / len(successful_results)
        avg_memory = sum(r.get('memory_mb', 0) for r in successful_results) / len(successful_results)
        
        print(f'   Avg Thoughts: {avg_thoughts:.0f}')
        print(f'   Avg Success Rate: {avg_success:.1%}') 
        print(f'   Avg Peak Memory: {avg_memory:.1f}MB')
        
        # Check variance
        thoughts = [r.get('thoughts', 0) for r in successful_results]
        success_rates = [r.get('success_rate', 0) for r in successful_results]
        
        if len(thoughts) > 1:
            thoughts_cv = (statistics.stdev(thoughts) / statistics.mean(thoughts)) * 100
            success_cv = (statistics.stdev(success_rates) / statistics.mean(success_rates)) * 100 if statistics.mean(success_rates) > 0 else 0
            
            print(f'   Thought Variance: CV={thoughts_cv:.1f}% ({"LOW" if thoughts_cv < 15 else "MODERATE" if thoughts_cv < 30 else "HIGH"})')
            print(f'   Success Rate Variance: CV={success_cv:.1f}% ({"LOW" if success_cv < 5 else "MODERATE" if success_cv < 15 else "HIGH"})')
    
    print(f'\n🏆 VERDICT:')
    if survived == len(results):
        print('🟢 EXCELLENT CONSISTENCY - All tests passed across chaos levels')
        print('   System demonstrates reliable resilience under extreme stress')
    elif survived >= len(results) * 0.8:
        print('🟡 GOOD CONSISTENCY - Most tests passed')
        print('   System is generally reliable but may have edge case vulnerabilities') 
    else:
        print('🔴 POOR CONSISTENCY - Multiple failures detected')
        print('   System has significant reliability issues under stress')

if __name__ == "__main__":
    asyncio.run(quick_validation())