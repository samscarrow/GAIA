#!/usr/bin/env python3
"""
Fault Injector - Controlled chaos for testing system resilience
"""

import os
import signal
import time
import threading
import random
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
import psutil
import subprocess

class FaultType(Enum):
    """Types of faults that can be injected"""
    PROCESS_KILL = "process_kill"          # kill -9
    PROCESS_STOP = "process_stop"          # SIGSTOP/SIGCONT
    DISK_FULL = "disk_full"                # Fill disk space
    SLOW_DISK = "slow_disk"                # Inject I/O delays
    NETWORK_JITTER = "network_jitter"      # Network delays/drops
    MEMORY_PRESSURE = "memory_pressure"     # Consume available memory
    CPU_SATURATION = "cpu_saturation"      # Max out CPU
    FILE_CORRUPTION = "file_corruption"     # Flip bits in files
    PERMISSION_DENIED = "permission_denied" # Remove file permissions
    CLOCK_SKEW = "clock_skew"              # System time changes

@dataclass
class FaultConfig:
    """Configuration for a specific fault"""
    fault_type: FaultType
    probability: float  # 0.0 to 1.0
    duration_seconds: float
    intensity: float  # 0.0 to 1.0
    target: Optional[str] = None  # Target file/process/etc
    enabled: bool = True

@dataclass
class FaultEvent:
    """Record of an injected fault"""
    fault_type: FaultType
    start_time: float
    end_time: Optional[float] = None
    target: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    
    @property 
    def duration(self) -> float:
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

class ProcessFaultInjector:
    """Inject process-related faults (kill, stop, etc.)"""
    
    def __init__(self):
        self.suspended_processes: List[int] = []
    
    def kill_process(self, pid: int) -> FaultEvent:
        """Kill process with SIGKILL"""
        event = FaultEvent(
            fault_type=FaultType.PROCESS_KILL,
            start_time=time.time(),
            target=str(pid)
        )
        
        try:
            process = psutil.Process(pid)
            process.kill()  # SIGKILL
            process.wait(timeout=5.0)
            event.success = True
            
        except (psutil.NoSuchProcess, psutil.TimeoutExpired) as e:
            event.error = str(e)
        except Exception as e:
            event.error = f"Unexpected error: {e}"
        
        event.end_time = time.time()
        return event
    
    def stop_resume_process(self, pid: int, duration: float) -> FaultEvent:
        """Stop process with SIGSTOP, then resume with SIGCONT"""
        event = FaultEvent(
            fault_type=FaultType.PROCESS_STOP,
            start_time=time.time(),
            target=str(pid)
        )
        
        try:
            process = psutil.Process(pid)
            
            # Stop the process
            process.send_signal(signal.SIGSTOP)
            self.suspended_processes.append(pid)
            
            # Wait for specified duration
            time.sleep(duration)
            
            # Resume the process
            process.send_signal(signal.SIGCONT)
            if pid in self.suspended_processes:
                self.suspended_processes.remove(pid)
                
            event.success = True
            
        except (psutil.NoSuchProcess, OSError) as e:
            event.error = str(e)
            if pid in self.suspended_processes:
                self.suspended_processes.remove(pid)
        except Exception as e:
            event.error = f"Unexpected error: {e}"
        
        event.end_time = time.time()
        return event
    
    def cleanup(self):
        """Resume any suspended processes"""
        for pid in self.suspended_processes[:]:
            try:
                process = psutil.Process(pid)
                process.send_signal(signal.SIGCONT)
                self.suspended_processes.remove(pid)
            except:
                pass

class DiskFaultInjector:
    """Inject disk-related faults"""
    
    def __init__(self):
        self.temp_files: List[str] = []
        self.original_permissions: Dict[str, int] = {}
    
    def fill_disk(self, target_path: str, fill_percent: float = 0.9) -> FaultEvent:
        """Fill disk to specified percentage"""
        event = FaultEvent(
            fault_type=FaultType.DISK_FULL,
            start_time=time.time(),
            target=target_path
        )
        
        try:
            # Get disk usage
            statvfs = os.statvfs(target_path)
            free_bytes = statvfs.f_bavail * statvfs.f_frsize
            total_bytes = statvfs.f_blocks * statvfs.f_frsize
            
            # Calculate how much to fill
            target_free = total_bytes * (1.0 - fill_percent)
            bytes_to_fill = free_bytes - target_free
            
            if bytes_to_fill > 0:
                # Create temporary file to fill space
                temp_file = os.path.join(target_path, f"fault_injector_fill_{int(time.time())}")
                
                with open(temp_file, 'wb') as f:
                    chunk_size = 1024 * 1024  # 1MB chunks
                    bytes_written = 0
                    
                    while bytes_written < bytes_to_fill:
                        chunk = b'0' * min(chunk_size, int(bytes_to_fill - bytes_written))
                        f.write(chunk)
                        bytes_written += len(chunk)
                        
                        # Don't completely fill disk - leave some room
                        if bytes_written > bytes_to_fill * 0.95:
                            break
                
                self.temp_files.append(temp_file)
                event.success = True
            else:
                event.error = f"Disk already at {fill_percent*100}% capacity"
            
        except OSError as e:
            event.error = f"Disk fill failed: {e}"
        except Exception as e:
            event.error = f"Unexpected error: {e}"
        
        event.end_time = time.time()
        return event
    
    def corrupt_file(self, file_path: str, corruption_rate: float = 0.01) -> FaultEvent:
        """Flip random bits in file"""
        event = FaultEvent(
            fault_type=FaultType.FILE_CORRUPTION,
            start_time=time.time(),
            target=file_path
        )
        
        try:
            if not os.path.exists(file_path):
                event.error = f"File not found: {file_path}"
                event.end_time = time.time()
                return event
            
            # Create backup first
            backup_path = f"{file_path}.fault_backup"
            shutil.copy2(file_path, backup_path)
            
            # Read file
            with open(file_path, 'rb') as f:
                data = bytearray(f.read())
            
            # Corrupt random bits
            num_bits = len(data) * 8
            num_corruptions = int(num_bits * corruption_rate)
            
            for _ in range(num_corruptions):
                byte_index = random.randint(0, len(data) - 1)
                bit_index = random.randint(0, 7)
                
                # Flip the bit
                data[byte_index] ^= (1 << bit_index)
            
            # Write corrupted data
            with open(file_path, 'wb') as f:
                f.write(data)
            
            event.success = True
            
        except Exception as e:
            event.error = f"File corruption failed: {e}"
        
        event.end_time = time.time()
        return event
    
    def remove_permissions(self, file_path: str) -> FaultEvent:
        """Remove read/write permissions from file"""
        event = FaultEvent(
            fault_type=FaultType.PERMISSION_DENIED,
            start_time=time.time(),
            target=file_path
        )
        
        try:
            if not os.path.exists(file_path):
                event.error = f"File not found: {file_path}"
            else:
                # Save original permissions
                stat_info = os.stat(file_path)
                self.original_permissions[file_path] = stat_info.st_mode
                
                # Remove all permissions
                os.chmod(file_path, 0o000)
                event.success = True
                
        except Exception as e:
            event.error = f"Permission change failed: {e}"
        
        event.end_time = time.time()
        return event
    
    def cleanup(self):
        """Clean up disk faults"""
        # Remove temp files
        for temp_file in self.temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        self.temp_files.clear()
        
        # Restore permissions
        for file_path, mode in self.original_permissions.items():
            try:
                os.chmod(file_path, mode)
            except:
                pass
        self.original_permissions.clear()

class ResourceExhaustionInjector:
    """Exhaust system resources (memory, CPU)"""
    
    def __init__(self):
        self.memory_hogs: List[Any] = []
        self.cpu_hogs: List[threading.Thread] = []
        self._stop_cpu_hogs = False
    
    def consume_memory(self, target_mb: int) -> FaultEvent:
        """Consume specified amount of memory"""
        event = FaultEvent(
            fault_type=FaultType.MEMORY_PRESSURE,
            start_time=time.time(),
            target=f"{target_mb}MB"
        )
        
        try:
            # Allocate memory in chunks
            chunk_size_mb = 100
            chunks = []
            
            for i in range(0, target_mb, chunk_size_mb):
                current_chunk = min(chunk_size_mb, target_mb - i)
                # Allocate and touch memory to force physical allocation
                chunk = bytearray(current_chunk * 1024 * 1024)
                for j in range(0, len(chunk), 4096):  # Touch every page
                    chunk[j] = 42
                chunks.append(chunk)
            
            self.memory_hogs.extend(chunks)
            event.success = True
            
        except MemoryError as e:
            event.error = f"Memory allocation failed: {e}"
        except Exception as e:
            event.error = f"Unexpected error: {e}"
        
        event.end_time = time.time()
        return event
    
    def saturate_cpu(self, num_threads: Optional[int] = None, duration: float = 30.0) -> FaultEvent:
        """Saturate CPU with busy loops"""
        if num_threads is None:
            num_threads = os.cpu_count() or 4
        
        event = FaultEvent(
            fault_type=FaultType.CPU_SATURATION,
            start_time=time.time(),
            target=f"{num_threads} threads"
        )
        
        try:
            self._stop_cpu_hogs = False
            
            def cpu_hog():
                """Busy loop to consume CPU"""
                while not self._stop_cpu_hogs:
                    # Busy work
                    sum(i * i for i in range(1000))
            
            # Start CPU hog threads
            threads = []
            for _ in range(num_threads):
                thread = threading.Thread(target=cpu_hog, daemon=True)
                thread.start()
                threads.append(thread)
            
            self.cpu_hogs.extend(threads)
            
            # Let them run for specified duration
            time.sleep(duration)
            
            # Stop CPU hogs
            self._stop_cpu_hogs = True
            
            # Wait for threads to finish
            for thread in threads:
                thread.join(timeout=1.0)
            
            event.success = True
            
        except Exception as e:
            event.error = f"CPU saturation failed: {e}"
        
        event.end_time = time.time()
        return event
    
    def cleanup(self):
        """Clean up resource consumption"""
        # Stop CPU hogs
        self._stop_cpu_hogs = True
        for thread in self.cpu_hogs:
            thread.join(timeout=1.0)
        self.cpu_hogs.clear()
        
        # Release memory
        self.memory_hogs.clear()

class NetworkFaultInjector:
    """Inject network-related faults"""
    
    def __init__(self):
        self.active_rules: List[str] = []
    
    def inject_latency(self, interface: str, delay_ms: int, jitter_ms: int = 0) -> FaultEvent:
        """Inject network latency using tc (Linux only)"""
        event = FaultEvent(
            fault_type=FaultType.NETWORK_JITTER,
            start_time=time.time(),
            target=f"{interface} +{delay_ms}ms"
        )
        
        try:
            # Check if we're on Linux and have tc command
            if os.name != 'posix':
                event.error = "Network injection only supported on Linux"
                event.end_time = time.time()
                return event
            
            # Build tc command
            cmd = [
                'tc', 'qdisc', 'add', 'dev', interface, 'root', 'netem',
                'delay', f'{delay_ms}ms'
            ]
            
            if jitter_ms > 0:
                cmd.extend([f'{jitter_ms}ms'])
            
            # Execute command
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            self.active_rules.append(interface)
            event.success = True
            
        except subprocess.CalledProcessError as e:
            event.error = f"tc command failed: {e.stderr}"
        except Exception as e:
            event.error = f"Network injection failed: {e}"
        
        event.end_time = time.time()
        return event
    
    def cleanup(self):
        """Remove network rules"""
        for interface in self.active_rules:
            try:
                subprocess.run(['tc', 'qdisc', 'del', 'dev', interface, 'root'], 
                             capture_output=True, check=False)
            except:
                pass
        self.active_rules.clear()

class FaultInjector:
    """Main fault injection orchestrator"""
    
    def __init__(self):
        self.process_injector = ProcessFaultInjector()
        self.disk_injector = DiskFaultInjector()
        self.resource_injector = ResourceExhaustionInjector()
        self.network_injector = NetworkFaultInjector()
        
        self.fault_events: List[FaultEvent] = []
        self.active_faults: List[FaultConfig] = []
        self._injection_thread: Optional[threading.Thread] = None
        self._stop_injection = False
    
    def register_fault(self, config: FaultConfig):
        """Register a fault for potential injection"""
        self.active_faults.append(config)
        print(f"🎯 Registered fault: {config.fault_type.value} (p={config.probability})")
    
    def inject_fault(self, fault_type: FaultType, target: Optional[str] = None, **kwargs) -> FaultEvent:
        """Inject a specific fault immediately"""
        print(f"💥 Injecting fault: {fault_type.value} on {target}")
        
        event = None
        
        if fault_type == FaultType.PROCESS_KILL:
            pid = int(target) if target else os.getpid()
            event = self.process_injector.kill_process(pid)
            
        elif fault_type == FaultType.PROCESS_STOP:
            pid = int(target) if target else os.getpid()
            duration = kwargs.get('duration', 5.0)
            event = self.process_injector.stop_resume_process(pid, duration)
            
        elif fault_type == FaultType.DISK_FULL:
            path = target or '/tmp'
            fill_pct = kwargs.get('fill_percent', 0.9)
            event = self.disk_injector.fill_disk(path, fill_pct)
            
        elif fault_type == FaultType.FILE_CORRUPTION:
            if not target:
                raise ValueError("File corruption requires target file path")
            corruption_rate = kwargs.get('corruption_rate', 0.01)
            event = self.disk_injector.corrupt_file(target, corruption_rate)
            
        elif fault_type == FaultType.PERMISSION_DENIED:
            if not target:
                raise ValueError("Permission fault requires target file path")
            event = self.disk_injector.remove_permissions(target)
            
        elif fault_type == FaultType.MEMORY_PRESSURE:
            memory_mb = kwargs.get('memory_mb', 1000)
            event = self.resource_injector.consume_memory(memory_mb)
            
        elif fault_type == FaultType.CPU_SATURATION:
            num_threads = kwargs.get('num_threads')
            duration = kwargs.get('duration', 30.0)
            event = self.resource_injector.saturate_cpu(num_threads, duration)
            
        elif fault_type == FaultType.NETWORK_JITTER:
            interface = target or 'lo'
            delay_ms = kwargs.get('delay_ms', 100)
            jitter_ms = kwargs.get('jitter_ms', 0)
            event = self.network_injector.inject_latency(interface, delay_ms, jitter_ms)
            
        else:
            event = FaultEvent(
                fault_type=fault_type,
                start_time=time.time(),
                target=target,
                error=f"Fault type not implemented: {fault_type.value}"
            )
            event.end_time = time.time()
        
        if event:
            self.fault_events.append(event)
            
            status = "✅" if event.success else "❌"
            print(f"   {status} Duration: {event.duration:.3f}s")
            if event.error:
                print(f"   Error: {event.error}")
        
        return event
    
    def start_continuous_injection(self, check_interval: float = 1.0):
        """Start continuous fault injection based on registered faults"""
        if self._injection_thread is not None:
            return
        
        self._stop_injection = False
        self._injection_thread = threading.Thread(target=self._injection_loop, 
                                                args=(check_interval,), daemon=True)
        self._injection_thread.start()
        print(f"🎲 Started continuous fault injection (interval: {check_interval}s)")
    
    def stop_continuous_injection(self):
        """Stop continuous fault injection"""
        self._stop_injection = True
        if self._injection_thread:
            self._injection_thread.join(timeout=5.0)
            self._injection_thread = None
        print("🛑 Stopped continuous fault injection")
    
    def _injection_loop(self, check_interval: float):
        """Main injection loop"""
        while not self._stop_injection:
            try:
                for config in self.active_faults:
                    if not config.enabled:
                        continue
                    
                    # Check probability
                    if random.random() < config.probability * check_interval:
                        self.inject_fault(config.fault_type, config.target,
                                        duration=config.duration_seconds,
                                        intensity=config.intensity)
                
                time.sleep(check_interval)
                
            except Exception as e:
                print(f"⚠️ Fault injection loop error: {e}")
                time.sleep(check_interval)
    
    @contextmanager
    def fault_context(self, fault_configs: List[FaultConfig]):
        """Context manager for fault injection during test execution"""
        # Register faults
        for config in fault_configs:
            self.register_fault(config)
        
        # Start injection
        self.start_continuous_injection(0.5)  # Check every 500ms
        
        try:
            yield self
        finally:
            # Clean up
            self.stop_continuous_injection()
            self.cleanup_all()
    
    def get_fault_summary(self) -> Dict[str, Any]:
        """Get summary of all injected faults"""
        total_faults = len(self.fault_events)
        successful_faults = sum(1 for event in self.fault_events if event.success)
        
        fault_counts = {}
        for event in self.fault_events:
            fault_type = event.fault_type.value
            fault_counts[fault_type] = fault_counts.get(fault_type, 0) + 1
        
        return {
            "total_faults_injected": total_faults,
            "successful_faults": successful_faults,
            "fault_success_rate": successful_faults / total_faults if total_faults > 0 else 0.0,
            "fault_breakdown": fault_counts,
            "registered_faults": len(self.active_faults)
        }
    
    def cleanup_all(self):
        """Clean up all fault injectors"""
        print("🧹 Cleaning up fault injectors...")
        try:
            self.process_injector.cleanup()
            self.disk_injector.cleanup()
            self.resource_injector.cleanup()
            self.network_injector.cleanup()
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

# Convenience functions for common fault scenarios
def chaos_monkey_config() -> List[FaultConfig]:
    """Standard chaos monkey fault configuration"""
    return [
        FaultConfig(FaultType.PROCESS_STOP, 0.001, 2.0, 0.5),  # 0.1% chance, 2s stop
        FaultConfig(FaultType.MEMORY_PRESSURE, 0.0001, 30.0, 0.7),  # 0.01% chance, consume memory
        FaultConfig(FaultType.CPU_SATURATION, 0.0001, 10.0, 0.8),  # 0.01% chance, saturate CPU
        FaultConfig(FaultType.DISK_FULL, 0.00001, 60.0, 0.9),  # Very rare, fill disk
    ]

def integration_test_faults() -> List[FaultConfig]:
    """Faults suitable for integration testing"""
    return [
        FaultConfig(FaultType.PROCESS_STOP, 0.01, 1.0, 0.3),  # 1% chance, 1s stop
        FaultConfig(FaultType.MEMORY_PRESSURE, 0.005, 15.0, 0.5),  # 0.5% chance
        FaultConfig(FaultType.NETWORK_JITTER, 0.02, 5.0, 0.4),  # 2% chance, 5s delay
    ]