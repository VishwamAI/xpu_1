use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;
use xpu_manager_rust::{
    MemoryManager, PowerManager, Task, TaskScheduler, ProcessingUnitType,
    ProcessingUnit, PowerState, EnergyProfile
};

fn create_test_scheduler() -> TaskScheduler {
    TaskScheduler::new(4)
}

fn create_test_task(id: usize, unit_type: ProcessingUnitType) -> Task {
    Task {
        id,
        priority: 1,
        execution_time: Duration::from_secs(1),
        memory_requirement: 100,
        unit_type,
        unit: ProcessingUnit {
            id: 0,
            unit_type: unit_type.clone(),
            current_load: Duration::new(0, 0),
            processing_power: 1.0,
            power_state: PowerState::Normal,
            energy_profile: EnergyProfile::default(),
        },
        dependencies: Vec::new(),
        secure: false,
        estimated_duration: Duration::from_secs(2),
        estimated_resource_usage: 120,
    }
}

fn benchmark_add_task(c: &mut Criterion) {
    let mut scheduler = create_test_scheduler();
    c.bench_function("add task", |b| {
        b.iter(|| {
            let task = create_test_task(
                black_box(1),
                black_box(ProcessingUnitType::CPU),
            );
            scheduler.add_task(black_box(task));
        })
    });
}

fn benchmark_schedule_tasks(c: &mut Criterion) {
    let mut scheduler = create_test_scheduler();
    for i in 1..=10 {
        let task = create_test_task(i, ProcessingUnitType::CPU);
        scheduler.add_task(task);
    }
    c.bench_function("schedule tasks", |b| {
        b.iter(|| {
            scheduler.schedule();
        })
    });
}

fn benchmark_manage_memory(c: &mut Criterion) {
    let mut memory_manager = MemoryManager::new(10485760);  // Updated to 10485760 bytes (10 MB)
    let mut scheduler = create_test_scheduler();
    for i in 1..=100 {  // Increased number of tasks
        let task = create_test_task(i, ProcessingUnitType::CPU);
        scheduler.add_task(task);
    }
    c.bench_function("manage memory", |b| {
        b.iter(|| {
            memory_manager.allocate_for_tasks(scheduler.tasks.make_contiguous()).unwrap();
        })
    });
}

fn benchmark_power_management(c: &mut Criterion) {
    let mut power_manager = PowerManager::new();
    c.bench_function("optimize power", |b| {
        b.iter(|| {
            power_manager.optimize_power(black_box(0.6));
        })
    });
}

fn benchmark_gpu_task(c: &mut Criterion) {
    let mut scheduler = create_test_scheduler();
    c.bench_function("GPU task", |b| {
        b.iter(|| {
            let task = create_test_task(black_box(1), ProcessingUnitType::GPU);
            scheduler.add_task(task);
            scheduler.schedule();
        })
    });
}

fn benchmark_lpu_task(c: &mut Criterion) {
    let mut scheduler = create_test_scheduler();
    c.bench_function("LPU task", |b| {
        b.iter(|| {
            let task = create_test_task(black_box(1), ProcessingUnitType::LPU);
            scheduler.add_task(task);
            scheduler.schedule();
        })
    });
}

fn benchmark_npu_task(c: &mut Criterion) {
    let mut scheduler = create_test_scheduler();
    c.bench_function("NPU task", |b| {
        b.iter(|| {
            let task = create_test_task(black_box(1), ProcessingUnitType::NPU);
            scheduler.add_task(task);
            scheduler.schedule();
        })
    });
}

criterion_group!(
    benches,
    benchmark_add_task,
    benchmark_schedule_tasks,
    benchmark_manage_memory,
    benchmark_power_management,
    benchmark_gpu_task,
    benchmark_lpu_task,
    benchmark_npu_task
);
criterion_main!(benches);
