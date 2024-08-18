use criterion::{black_box, criterion_group, criterion_main, Criterion};
use xpu_optimization::{XpuOptimizer, XpuOptimizerConfig, SchedulerType, MemoryManagerType, Task, ProcessingUnit, ProcessingUnitType};
use std::time::Duration;

fn create_test_optimizer() -> XpuOptimizer {
    let config = XpuOptimizerConfig {
        num_processing_units: 4,
        memory_pool_size: 1024,
        scheduler_type: SchedulerType::RoundRobin,
        memory_manager_type: MemoryManagerType::Simple,
    };
    XpuOptimizer::new(config).unwrap()
}

fn create_test_task(id: usize) -> Task {
    Task {
        id,
        unit: ProcessingUnit {
            unit_type: ProcessingUnitType::CPU,
            processing_power: 1.0,
            current_load: 0.0,
        },
        priority: 1,
        dependencies: vec![],
        execution_time: Duration::from_secs(1),
        memory_requirement: 100,
    }
}

fn benchmark_add_task(c: &mut Criterion) {
    let mut optimizer = create_test_optimizer();
    c.bench_function("add task", |b| {
        b.iter(|| {
            let task = create_test_task(black_box(1));
            optimizer.add_task(task).unwrap();
        })
    });
}

fn benchmark_schedule_tasks(c: &mut Criterion) {
    let mut optimizer = create_test_optimizer();
    for i in 1..=10 {
        let task = create_test_task(i);
        optimizer.add_task(task).unwrap();
    }
    c.bench_function("schedule tasks", |b| {
        b.iter(|| {
            optimizer.schedule_tasks().unwrap();
        })
    });
}

fn benchmark_manage_memory(c: &mut Criterion) {
    let mut optimizer = create_test_optimizer();
    for i in 1..=10 {
        let task = create_test_task(i);
        optimizer.add_task(task).unwrap();
    }
    c.bench_function("manage memory", |b| {
        b.iter(|| {
            optimizer.manage_memory().unwrap();
        })
    });
}

criterion_group!(benches, benchmark_add_task, benchmark_schedule_tasks, benchmark_manage_memory);
criterion_main!(benches);
