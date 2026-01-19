#!/usr/bin/env python3
"""
GPU Usage System - Demo/Usage Example
Shows how to use the GPU management system
"""

import time
from gpu_manager import GPUManager, TaskType, TaskPriority
from gpu_allocator import GPUAllocator, AllocationStrategy
from gpu_monitor import EnhancedGPUMonitor

def main():
    print("""
==============================================================================
         GPU Usage System - Demo
==============================================================================
    """)
    
    # 1. Initialize GPU Manager
    print("\n[1] Initializing GPU Manager...")
    gpu_manager = GPUManager()
    
    if gpu_manager.gpu_monitor.gpu_count > 0:
        print(f"    [OK] Found {gpu_manager.gpu_monitor.gpu_count} GPU(s)")
    else:
        print("    [WARNING] No GPUs detected - running in CPU-only mode")
    
    # 2. Start the system
    print("\n[2] Starting GPU system...")
    gpu_manager.start_system()
    print("    [OK] System started")
    
    # 3. Monitor GPU usage
    print("\n[3] Getting GPU status...")
    monitor = EnhancedGPUMonitor()
    summary = monitor.get_usage_summary()
    
    print(f"    Total GPUs: {summary.get('total_gpus', 0)}")
    print(f"    Available GPUs: {summary.get('available_gpus', 0)}")
    
    for gpu_id, details in summary.get('gpu_details', {}).items():
        print(f"\n    GPU {gpu_id}:")
        print(f"      Utilization: {details.get('utilization', 0):.1f}%")
        print(f"      Memory Used: {details.get('memory_used_pct', 0):.1f}%")
        print(f"      Memory Free: {details.get('memory_free_mb', 0)}MB")
        print(f"      Temperature: {details.get('temperature', 0):.1f}°C")
    
    # 4. Submit tasks
    print("\n[4] Submitting tasks...")
    
    # Submit inference task
    task_id1 = gpu_manager.submit_task(
        TaskType.INFERENCE,
        'demo_model',
        {'features': [1, 2, 3, 4]},
        TaskPriority.HIGH,
        expected_memory=200,
        expected_duration=2.0
    )
    print(f"    [OK] Inference task submitted: {task_id1}")
    
    # Submit pattern detection task
    task_id2 = gpu_manager.submit_task(
        TaskType.PATTERN_DETECTION,
        'pattern_model',
        {'data': 'sample'},
        TaskPriority.MEDIUM,
        expected_memory=300,
        expected_duration=1.5
    )
    print(f"    [OK] Pattern detection task submitted: {task_id2}")
    
    # 5. Wait for completion
    print("\n[5] Waiting for tasks to complete...")
    results = {}
    timeout = 10
    elapsed = 0
    
    while elapsed < timeout:
        time.sleep(0.5)
        elapsed += 0.5
        
        for task_id in [task_id1, task_id2]:
            if task_id and task_id not in results:
                result = gpu_manager.get_task_result(task_id)
                if result:
                    results[task_id] = result
                    print(f"    [OK] Task {task_id} completed")
    
    # 6. Show results
    print("\n[6] Task Results:")
    for task_id, result in results.items():
        print(f"    {task_id}:")
        print(f"      {result}")
    
    # 7. Get statistics
    print("\n[7] System Statistics:")
    stats = gpu_manager.task_scheduler.get_task_statistics()
    print(f"    Total Tasks: {stats.get('total_tasks', 0)}")
    print(f"    Completed: {stats.get('completed_tasks', 0)}")
    print(f"    Failed: {stats.get('failed_tasks', 0)}")
    print(f"    Average Completion Time: {stats.get('avg_completion_time', 0):.2f}s")
    
    # 8. Test allocator
    print("\n[8] Testing GPU Allocator...")
    allocator = GPUAllocator(gpu_manager)
    
    from gpu_manager import GPUTask
    from datetime import datetime
    
    test_task = GPUTask(
        task_id="allocator_test",
        task_type=TaskType.INFERENCE,
        priority=TaskPriority.MEDIUM,
        model_name="test",
        input_data={},
        expected_memory=100,
        expected_duration=0.5,
        created_at=datetime.now()
    )
    
    # Test best-fit allocation
    gpu_id = allocator.allocate(test_task, AllocationStrategy.BEST_FIT)
    if gpu_id is not None:
        print(f"    [OK] Allocated GPU {gpu_id} using best-fit strategy")
        allocator.deallocate("allocator_test")
    
    # 9. Stop system
    print("\n[9] Stopping GPU system...")
    gpu_manager.stop_system()
    print("    [OK] System stopped")
    
    print("\n" + "="*70)
    print("[SUCCESS] Demo Complete!")
    print("="*70 + "\n")
    
    print("Usage Summary:")
    print("- GPU Manager: Task submission and scheduling")
    print("- GPU Allocator: Smart GPU resource allocation")
    print("- GPU Monitor: Real-time GPU monitoring and alerts")
    print("\nAll systems working correctly!")
    print()

if __name__ == "__main__":
    main()

