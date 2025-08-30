#!/usr/bin/env python3
"""
Quick Breaking Point Test - Find GAIA's actual limits in under 60 seconds
"""

import asyncio
import sys
import os
import time
import psutil
import numpy as np
import threading
sys.path.append('.')

from kernel.enhanced_core import EnhancedGAIAKernel

async def quick_memory_break():
    """Try to exhaust memory in GAIA system"""
    print("💣 MEMORY BREAK TEST")
    
    # Start with tiny memory to force breaking faster  
    kernel = EnhancedGAIAKernel(memory_size_mb=4)
    await kernel.initialize()
    
    bombs = []
    mb_allocated = 0
    
    try:
        for i in range(200):  # Try 200 x 10MB = 2GB
            # Allocate 10MB chunks
            bomb = np.ones((2560 * 1024,), dtype=np.float32)  # 10MB
            bomb.fill(i)  # Force physical allocation
            bombs.append(bomb)
            mb_allocated += 10
            
            if i % 10 == 0:
                process = psutil.Process()
                print(f"  {mb_allocated}MB allocated, RSS: {process.memory_info().rss//1024//1024}MB")
            
            # Try to use kernel with memory pressure
            try:
                await kernel.memory_manager.store(
                    f"break_test_{i}",
                    np.random.randn(100).astype(np.float32),
                    semantic_category="break_test"
                )
            except Exception as e:
                print(f"💥 KERNEL BROKE at {mb_allocated}MB: {e}")
                return mb_allocated, "KERNEL_ERROR", str(e)
            
            await asyncio.sleep(0.01)
            
    except MemoryError as e:
        print(f"💥 MEMORY EXHAUSTION at {mb_allocated}MB")
        return mb_allocated, "MEMORY_ERROR", str(e)
    except Exception as e:
        print(f"💥 SYSTEM ERROR at {mb_allocated}MB: {e}")
        return mb_allocated, "SYSTEM_ERROR", str(e)
    
    print(f"😳 No breaking point found after {mb_allocated}MB")
    return mb_allocated, "NO_BREAK", "System survived unrealistic load"

def thread_bomb_test():
    """Find thread creation limit"""
    print("\n🧵 THREAD BOMB TEST")
    
    threads = []
    
    def worker():
        time.sleep(30)  # Keep thread alive
    
    try:
        for i in range(10000):
            try:
                thread = threading.Thread(target=worker)
                thread.daemon = True
                thread.start()
                threads.append(thread)
                
                if i % 500 == 0:
                    print(f"  Created {i} threads")
                    
            except Exception as e:
                print(f"💥 THREAD LIMIT at {i} threads: {e}")
                return i, "THREAD_LIMIT", str(e)
    
    except Exception as e:
        print(f"💥 THREAD ERROR: {e}")
        return len(threads), "THREAD_ERROR", str(e)
    
    print(f"😳 Created {len(threads)} threads without breaking")
    return len(threads), "NO_BREAK", "System survived thread bomb"

def recursion_bomb_test():
    """Force stack overflow"""
    print("\n📚 RECURSION BOMB TEST") 
    
    depth = 0
    
    try:
        def recurse(d):
            nonlocal depth
            depth = d
            # Allocate stack space
            local_vars = [i for i in range(50)]
            local_string = f"depth_{d}_" + "x" * 50
            
            if d % 500 == 0:
                print(f"  Recursion depth: {d}")
                
            return recurse(d + 1)
            
        recurse(0)
        
    except RecursionError as e:
        print(f"💥 STACK OVERFLOW at depth {depth}")
        return depth, "STACK_OVERFLOW", str(e)
    except Exception as e:
        print(f"💥 RECURSION ERROR at depth {depth}: {e}")
        return depth, "RECURSION_ERROR", str(e)
    
    print(f"😳 No stack overflow at depth {depth}")
    return depth, "NO_BREAK", "No recursion limit hit"

async def run_quick_break_tests():
    """Run all breaking point tests quickly"""
    print("🔥 QUICK BREAKING POINT TESTS")
    print("=" * 50)
    
    results = {}
    
    # Test 1: Memory exhaustion
    try:
        mem_limit, mem_type, mem_error = await quick_memory_break()
        results['memory'] = {'limit': mem_limit, 'type': mem_type, 'error': mem_error}
    except Exception as e:
        results['memory'] = {'limit': 0, 'type': 'TEST_ERROR', 'error': str(e)}
    
    # Test 2: Thread limit
    try:
        thread_limit, thread_type, thread_error = thread_bomb_test()
        results['threads'] = {'limit': thread_limit, 'type': thread_type, 'error': thread_error}
    except Exception as e:
        results['threads'] = {'limit': 0, 'type': 'TEST_ERROR', 'error': str(e)}
    
    # Test 3: Stack overflow
    try:
        stack_limit, stack_type, stack_error = recursion_bomb_test()  
        results['stack'] = {'limit': stack_limit, 'type': stack_type, 'error': stack_error}
    except Exception as e:
        results['stack'] = {'limit': 0, 'type': 'TEST_ERROR', 'error': str(e)}
    
    # Report
    print(f"\n📊 BREAKING POINT RESULTS:")
    print("=" * 50)
    
    breaking_points_found = 0
    
    for test_name, result in results.items():
        limit_type = result['type']
        
        if limit_type in ['KERNEL_ERROR', 'MEMORY_ERROR', 'THREAD_LIMIT', 'STACK_OVERFLOW']:
            status = f"💥 BROKE at {result['limit']}"
            breaking_points_found += 1
        elif limit_type == 'NO_BREAK':
            status = f"😳 SURVIVED {result['limit']}"
        else:
            status = f"❌ TEST FAILED: {result['error'][:50]}..."
            
        print(f"  {test_name.upper()}: {status}")
        print(f"    Type: {limit_type}")
        print(f"    Details: {result['error'][:100]}...")
    
    print(f"\n🏆 ASSESSMENT:")
    if breaking_points_found == 0:
        print("🔴 CRITICAL: NO BREAKING POINTS FOUND")
        print("   Either system is impossibly robust or tests are too weak")
        print("   Real systems MUST have limits - investigate test validity")
    elif breaking_points_found == len(results):
        print("🟢 EXCELLENT: All systems have measurable limits")
        print("   System behaves predictably under extreme conditions")
    else:
        print(f"🟡 MIXED: {breaking_points_found}/{len(results)} systems found limits")
        print("   Some subsystems may need stronger testing")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_quick_break_tests())