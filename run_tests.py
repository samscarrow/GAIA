#!/usr/bin/env python3
"""
Automated test runner for GAIA architecture with comprehensive reporting
"""

import asyncio
import subprocess
import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import argparse

class TestSuite(Enum):
    """Available test suites"""
    UNIT = "unit"
    INTEGRATION = "integration"
    STRESS = "stress"
    SIMULATION = "simulation"
    PERFORMANCE = "performance"
    ALL = "all"

@dataclass
class TestResult:
    """Individual test result"""
    suite: str
    test_file: str
    status: str  # passed, failed, error, skipped
    duration: float
    output: str
    error: Optional[str] = None

@dataclass
class TestReport:
    """Comprehensive test report"""
    timestamp: str
    version: str
    total_duration: float
    summary: Dict[str, int]  # passed, failed, error, skipped counts
    results: List[TestResult]
    system_info: Dict[str, Any]
    coverage_info: Optional[Dict[str, Any]] = None

class TestRunner:
    """Orchestrates all GAIA tests with reporting"""
    
    def __init__(self, verbose: bool = False, parallel: bool = True):
        self.verbose = verbose
        self.parallel = parallel
        self.results: List[TestResult] = []
        
        # Test suite configurations
        self.test_configs = {
            TestSuite.UNIT: {
                "files": [
                    "tests/unit/test_memory_zones.py",
                ],
                "description": "Unit tests for core components",
                "timeout": 120,
                "critical": True
            },
            TestSuite.INTEGRATION: {
                "files": [
                    "tests/integration/test_kernel_integration.py",
                ],
                "description": "Integration tests for component interactions",
                "timeout": 180,
                "critical": True
            },
            TestSuite.STRESS: {
                "files": [
                    "tests/stress/test_extreme_conditions.py",
                ],
                "description": "Stress tests for extreme conditions",
                "timeout": 300,
                "critical": False
            },
            TestSuite.SIMULATION: {
                "files": [
                    "tests/simulation/cognitive_workloads.py",
                ],
                "description": "Real-world cognitive workload simulation",
                "timeout": 600,
                "critical": False
            },
            TestSuite.PERFORMANCE: {
                "files": [
                    "tests/performance/regression_tests.py",
                ],
                "description": "Performance regression tests",
                "timeout": 300,
                "critical": False
            }
        }
    
    def print_banner(self):
        """Print test runner banner"""
        print("🧪 GAIA ARCHITECTURE TEST SUITE")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Verbose: {self.verbose}")
        print(f"Parallel: {self.parallel}")
        print("=" * 60)
    
    async def run_test_suite(self, suite: TestSuite, **kwargs) -> List[TestResult]:
        """Run a specific test suite"""
        if suite == TestSuite.ALL:
            return await self.run_all_suites(**kwargs)
        
        if suite not in self.test_configs:
            raise ValueError(f"Unknown test suite: {suite}")
        
        config = self.test_configs[suite]
        print(f"\n🚀 Running {suite.value} tests: {config['description']}")
        
        suite_results = []
        
        if self.parallel and len(config["files"]) > 1:
            # Run files in parallel
            tasks = []
            for test_file in config["files"]:
                task = asyncio.create_task(
                    self._run_test_file(test_file, suite.value, config["timeout"], **kwargs)
                )
                tasks.append(task)
            
            suite_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            for i, result in enumerate(suite_results):
                if isinstance(result, Exception):
                    suite_results[i] = TestResult(
                        suite=suite.value,
                        test_file=config["files"][i],
                        status="error",
                        duration=0,
                        output="",
                        error=str(result)
                    )
        else:
            # Run files sequentially
            for test_file in config["files"]:
                result = await self._run_test_file(
                    test_file, suite.value, config["timeout"], **kwargs
                )
                suite_results.append(result)
        
        return suite_results
    
    async def run_all_suites(self, **kwargs) -> List[TestResult]:
        """Run all test suites"""
        all_results = []
        
        # Define order (critical tests first)
        suite_order = [
            TestSuite.UNIT,
            TestSuite.INTEGRATION,
            TestSuite.PERFORMANCE,
            TestSuite.STRESS,
            TestSuite.SIMULATION
        ]
        
        for suite in suite_order:
            try:
                results = await self.run_test_suite(suite, **kwargs)
                all_results.extend(results)
                
                # Check if critical suite failed
                if self.test_configs[suite]["critical"]:
                    failed_critical = any(r.status in ["failed", "error"] for r in results)
                    if failed_critical:
                        print(f"❌ Critical test suite {suite.value} failed - stopping execution")
                        break
                        
            except Exception as e:
                print(f"💥 Error running {suite.value} suite: {e}")
                all_results.append(TestResult(
                    suite=suite.value,
                    test_file="suite",
                    status="error",
                    duration=0,
                    output="",
                    error=str(e)
                ))
                
                if self.test_configs[suite]["critical"]:
                    break
        
        return all_results
    
    async def _run_test_file(self, test_file: str, suite_name: str, 
                            timeout: int, **kwargs) -> TestResult:
        """Run a single test file"""
        print(f"  📄 Running {test_file}...")
        
        start_time = time.time()
        
        try:
            # Prepare command
            if test_file.endswith('.py') and 'pytest' in test_file or '/unit/' in test_file or '/integration/' in test_file or '/stress/' in test_file:
                # Use pytest for test files
                cmd = [sys.executable, "-m", "pytest", test_file, "-v"]
                if kwargs.get('coverage'):
                    cmd.extend(["--cov=.", "--cov-report=json"])
            else:
                # Direct execution for simulation and performance files
                cmd = [sys.executable, test_file]
                
                # Add any additional arguments
                if suite_name == "performance":
                    if kwargs.get('baseline'):
                        cmd.append("--baseline")
                    if kwargs.get('version'):
                        cmd.append(kwargs['version'])
            
            # Run command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                
                duration = time.time() - start_time
                
                # Determine status
                if process.returncode == 0:
                    status = "passed"
                    error = None
                else:
                    status = "failed"
                    error = stderr.decode() if stderr else "Process failed"
                
                # Get output
                output = stdout.decode() if stdout else ""
                if self.verbose:
                    print(f"    Output: {output[:200]}...")
                
                print(f"    ✅ {status.upper()} in {duration:.2f}s")
                
                return TestResult(
                    suite=suite_name,
                    test_file=test_file,
                    status=status,
                    duration=duration,
                    output=output,
                    error=error
                )
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                
                duration = time.time() - start_time
                print(f"    ⏰ TIMEOUT after {duration:.2f}s")
                
                return TestResult(
                    suite=suite_name,
                    test_file=test_file,
                    status="error",
                    duration=duration,
                    output="",
                    error=f"Timeout after {timeout}s"
                )
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"    💥 ERROR: {e}")
            
            return TestResult(
                suite=suite_name,
                test_file=test_file,
                status="error",
                duration=duration,
                output="",
                error=str(e)
            )
    
    def generate_report(self, results: List[TestResult], version: str = "current") -> TestReport:
        """Generate comprehensive test report"""
        total_duration = sum(r.duration for r in results)
        
        # Count results by status
        summary = {
            "passed": sum(1 for r in results if r.status == "passed"),
            "failed": sum(1 for r in results if r.status == "failed"),
            "error": sum(1 for r in results if r.status == "error"),
            "skipped": sum(1 for r in results if r.status == "skipped")
        }
        
        # System info
        system_info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "cwd": os.getcwd(),
            "test_runner_version": "1.0.0"
        }
        
        return TestReport(
            timestamp=datetime.now().isoformat(),
            version=version,
            total_duration=total_duration,
            summary=summary,
            results=results,
            system_info=system_info
        )
    
    def print_report(self, report: TestReport):
        """Print human-readable test report"""
        print(f"\n📊 TEST EXECUTION REPORT")
        print("=" * 60)
        print(f"Version: {report.version}")
        print(f"Duration: {report.total_duration:.2f}s")
        print(f"Total tests: {len(report.results)}")
        
        # Summary
        print(f"\nSUMMARY:")
        print(f"  ✅ Passed:  {report.summary['passed']}")
        print(f"  ❌ Failed:  {report.summary['failed']}")
        print(f"  💥 Errors:  {report.summary['error']}")
        print(f"  ⏭️  Skipped: {report.summary['skipped']}")
        
        # Success rate
        total_tests = len(report.results)
        success_rate = report.summary['passed'] / total_tests if total_tests > 0 else 0
        print(f"  📈 Success rate: {success_rate:.1%}")
        
        # Detailed results
        if report.summary['failed'] > 0 or report.summary['error'] > 0:
            print(f"\nFAILED/ERROR DETAILS:")
            for result in report.results:
                if result.status in ["failed", "error"]:
                    print(f"  💥 {result.suite}/{result.test_file}: {result.status}")
                    if result.error and self.verbose:
                        print(f"     Error: {result.error[:200]}...")
        
        # Performance summary
        suite_performance = {}
        for result in report.results:
            if result.suite not in suite_performance:
                suite_performance[result.suite] = []
            suite_performance[result.suite].append(result.duration)
        
        print(f"\nPERFORMANCE BY SUITE:")
        for suite, durations in suite_performance.items():
            avg_duration = sum(durations) / len(durations)
            print(f"  {suite}: {avg_duration:.2f}s avg ({min(durations):.2f}s - {max(durations):.2f}s)")
        
        # Final verdict
        print(f"\n🏆 FINAL VERDICT:")
        if report.summary['error'] > 0:
            print("🔴 CRITICAL ERRORS - System has serious issues")
        elif report.summary['failed'] > 0:
            print("🟡 SOME FAILURES - Review failed tests")
        elif success_rate >= 0.95:
            print("🟢 EXCELLENT - All systems operational")
        else:
            print("🟡 ACCEPTABLE - Most tests passing")
    
    def save_report(self, report: TestReport, filename: Optional[str] = None):
        """Save report to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_report_{timestamp}.json"
        
        # Convert report to dict for JSON serialization
        report_dict = {
            "timestamp": report.timestamp,
            "version": report.version,
            "total_duration": report.total_duration,
            "summary": report.summary,
            "results": [
                {
                    "suite": r.suite,
                    "test_file": r.test_file,
                    "status": r.status,
                    "duration": r.duration,
                    "output": r.output[:1000] if r.output else "",  # Limit output size
                    "error": r.error
                } for r in report.results
            ],
            "system_info": report.system_info
        }
        
        with open(filename, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"💾 Report saved to {filename}")

async def main():
    """Main test runner entry point"""
    parser = argparse.ArgumentParser(description="GAIA Architecture Test Runner")
    parser.add_argument("suite", nargs='?', default="all", 
                       choices=[s.value for s in TestSuite],
                       help="Test suite to run")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="Run tests in parallel")
    parser.add_argument("--version", default="current",
                       help="Version identifier for testing")
    parser.add_argument("--baseline", action="store_true",
                       help="Save performance results as baseline")
    parser.add_argument("--coverage", action="store_true",
                       help="Generate coverage report")
    parser.add_argument("--report", help="Save report to specific file")
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner(verbose=args.verbose, parallel=args.parallel)
    runner.print_banner()
    
    try:
        # Run tests
        suite = TestSuite(args.suite)
        results = await runner.run_test_suite(
            suite, 
            version=args.version,
            baseline=args.baseline,
            coverage=args.coverage
        )
        
        # Generate and print report
        report = runner.generate_report(results, args.version)
        runner.print_report(report)
        
        # Save report if requested
        if args.report:
            runner.save_report(report, args.report)
        
        # Exit with appropriate code
        if report.summary['error'] > 0 or report.summary['failed'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⛔ Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())