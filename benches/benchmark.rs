use criterion::{black_box, criterion_group, criterion_main, Criterion};
use xpu_manager_rust::{TaskScheduler, Task, MemoryManager, PowerManager};
use std::time::Duration;

fn create_test_scheduler() -> TaskScheduler {
    TaskScheduler::new(4)
}

fn create_test_task(id: usize) -> Task {
    Task {
        id,
        priority: 1,
        execution_time: Duration::from_secs(1),
        memory_requirement: 100,
    }
}

fn benchmark_add_task(c: &mut Criterion) {
    let mut scheduler = create_test_scheduler();
    c.bench_function("add task", |b| {
        b.iter(|| {
            let task = create_test_task(black_box(1));
            scheduler.add_task(task);
        })
    });
}

fn benchmark_schedule_tasks(c: &mut Criterion) {
    let mut scheduler = create_test_scheduler();
    for i in 1..=10 {
        let task = create_test_task(i);
        scheduler.add_task(task);
    }
    c.bench_function("schedule tasks", |b| {
        b.iter(|| {
            scheduler.schedule();
        })
    });
}

fn benchmark_manage_memory(c: &mut Criterion) {
    let mut memory_manager = MemoryManager::new(1024);
    let mut scheduler = create_test_scheduler();
    for i in 1..=10 {
        let task = create_test_task(i);
        scheduler.add_task(task);
    }
    c.bench_function("manage memory", |b| {
        b.iter(|| {
            memory_manager.allocate_for_tasks(&scheduler.tasks).unwrap();
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

criterion_group!(benches, benchmark_add_task, benchmark_schedule_tasks, benchmark_manage_memory, benchmark_power_management);
criterion_main!(benches);
