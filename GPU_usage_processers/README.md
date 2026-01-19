# GPU Management System

## Overview

Intelligent GPU resource management system for distributing AI training and inference workloads across multiple GPUs.

## Features

- ✅ Multi-GPU support
- ✅ Automatic load balancing
- ✅ GPU resource monitoring
- ✅ Dynamic allocation
- ✅ Task scheduling
- ✅ Memory optimization
- ✅ Real-time monitoring with alerts
- ✅ Works without GPUtil (fallback mode)

## Quick Start

### Run Tests

```bash
cd GPU_usage_processers
python test_gpu_system.py
```

### Run Demo

```bash
python demo_usage.py
```

## Files

| File | Description |
|------|-------------|
| `gpu_manager.py` | Main GPU management system |
| `gpu_monitor.py` | GPU monitoring with alerts |
| `gpu_allocator.py` | Smart GPU allocation |
| `config.py` | Configuration settings |
| `test_gpu_system.py` | Test suite |
| `demo_usage.py` | Usage examples |

## Usage Examples

### 1. Basic Usage - Task Submission

```python
from gpu_manager import GPUManager, TaskType, TaskPriority

# Initialize GPU Manager
gpu_manager = GPUManager()

# Start the system
gpu_manager.start_system()

# Submit a task
task_id = gpu_manager.submit_task(
    TaskType.INFERENCE,
    'model_name',
    {'input': 'data'},
    TaskPriority.HIGH,
    expected_memory=200,  # MB
    expected_duration=5.0  # seconds
)

# Wait for result
import time
while not gpu_manager.get_task_result(task_id):
    time.sleep(0.5)

result = gpu_manager.get_task_result(task_id)
print(f"Result: {result}")

# Stop system
gpu_manager.stop_system()
```

### 2. GPU Monitoring

```python
from gpu_monitor import EnhancedGPUMonitor

monitor = EnhancedGPUMonitor()

# Get usage summary
summary = monitor.get_usage_summary()
print(f"Total GPUs: {summary['total_gpus']}")
print(f"Available GPUs: {summary['available_gpus']}")

# Get GPU details
for gpu_id, details in summary['gpu_details'].items():
    print(f"GPU {gpu_id}:")
    print(f"  Utilization: {details['utilization']}%")
    print(f"  Memory Used: {details['memory_used_pct']}%")
    print(f"  Temperature: {details['temperature']}°C")

# Register alert callback
def handle_alert(alert):
    print(f"Alert: {alert.message}")

monitor.register_alert_callback(handle_alert)
```

### 3. Smart GPU Allocation

```python
from gpu_allocator import GPUAllocator, AllocationStrategy
from gpu_manager import GPUManager, GPUTask, TaskType, TaskPriority
from datetime import datetime

gpu_manager = GPUManager()
gpu_manager.start_system()

allocator = GPUAllocator(gpu_manager)

# Create a task
task = GPUTask(
    task_id="my_task",
    task_type=TaskType.TRAINING,
    priority=TaskPriority.HIGH,
    model_name="model",
    input_data={},
    expected_memory=1000,
    expected_duration=300.0,
    created_at=datetime.now()
)

# Allocate GPU using different strategies
strategies = [
    AllocationStrategy.BEST_FIT,
    AllocationStrategy.FIRST_FIT,
    AllocationStrategy.ROUND_ROBIN,
    AllocationStrategy.LOAD_BALANCED
]

for strategy in strategies:
    gpu_id = allocator.allocate(task, strategy)
    if gpu_id is not None:
        print(f"Allocated GPU {gpu_id} using {strategy.value}")
        allocator.deallocate(task.task_id)
```

### 4. Task Types and Priorities

```python
from gpu_manager import GPUManager, TaskType, TaskPriority

gpu_manager = GPUManager()
gpu_manager.start_system()

# Different task types
task_types = [
    TaskType.TRAINING,
    TaskType.INFERENCE,
    TaskType.PATTERN_DETECTION,
    TaskType.SENTIMENT_ANALYSIS
]

# Different priorities
priorities = [
    TaskPriority.LOW,
    TaskPriority.MEDIUM,
    TaskPriority.HIGH,
    TaskPriority.CRITICAL
]

# Submit high-priority inference task
task_id = gpu_manager.submit_task(
    TaskType.INFERENCE,
    'model',
    {'data': 'input'},
    TaskPriority.CRITICAL,
    expected_memory=500,
    expected_duration=2.0
)
```

## Configuration

Edit `config.py` to customize:

```python
CONFIG = {
    'max_memory_usage': 0.9,  # 90% max memory
    'max_concurrent_tasks': 4,
    'monitoring_interval_seconds': 5,
    'temperature_threshold_celsius': 80,
    'task_scheduling_strategy': 'priority',
    'gpu_selection_strategy': 'best_fit'
}
```

## Running Tests

```bash
# Run all tests
python test_gpu_system.py

# Run demo
python demo_usage.py
```

## Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- `torch>=2.0.0` - PyTorch (optional, works in CPU mode)
- `GPUtil>=1.4.0` - GPU monitoring (optional)
- `psutil>=5.9.0` - System monitoring (optional)

The system works even without these packages - it runs in fallback mode.

## Features

### Automatic Resource Management
- Monitors GPU memory and utilization
- Allocates GPUs based on task requirements
- Balances load across multiple GPUs

### Smart Scheduling
- Priority-based task scheduling
- Multiple allocation strategies
- Automatic task execution

### Monitoring & Alerts
- Real-time GPU monitoring
- Temperature and memory alerts
- Performance tracking
- Export monitoring data

## Test Results

```
[OK] GPU Monitor: PASSED
[OK] GPU Allocator: PASSED
[OK] GPU Manager: PASSED
```

The system detects and manages GPUs automatically!

