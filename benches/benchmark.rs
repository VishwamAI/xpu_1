use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;
use xpu_manager_rust::{
    memory_management::MemoryManager,
    power_management::{EnergyProfile, PowerManager, PowerState},
    task_scheduling::{ProcessingUnit, ProcessingUnitType, Task, TaskScheduler},
};

fn create_test_scheduler() -> TaskScheduler {
    TaskScheduler::new(4)
}

fn create_test_task(id: usize, unit_type: ProcessingUnitType) -> Task {
    let cloned_unit_type = unit_type.clone();
    Task {
        id,
        priority: 1,
        execution_time: Duration::from_secs(1),
        memory_requirement: 10, // Reduced from 100 to 10
        unit_type: cloned_unit_type.clone(),
        unit: ProcessingUnit {
            id: 0,
            unit_type: cloned_unit_type,
            current_load: Duration::new(0, 0),
            processing_power: 1.0,
            power_state: PowerState::Normal,
            energy_profile: EnergyProfile::default(),
        },
        dependencies: Vec::new(),
        secure: false,
        estimated_duration: Duration::from_secs(2),
        estimated_resource_usage: 12, // Reduced from 120 to 12
    }
}

fn benchmark_add_task(c: &mut Criterion) {
    let mut scheduler = create_test_scheduler();
    c.bench_function("add task", |b| {
        b.iter(|| {
            let task = create_test_task(black_box(1), black_box(ProcessingUnitType::CPU));
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
            let _ = scheduler.schedule();
        })
    });
}

fn benchmark_manage_memory(c: &mut Criterion) {
    let mut memory_manager = MemoryManager::new(1048576); // 1 MB
    let mut scheduler = create_test_scheduler();
    for i in 1..=20 {
        // Reduced number of tasks
        let task = create_test_task(i, ProcessingUnitType::CPU);
        scheduler.add_task(task);
    }
    c.bench_function("manage memory", |b| {
        b.iter(|| {
            memory_manager
                .allocate_for_tasks(scheduler.tasks.make_contiguous())
                .unwrap();
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
            let _ = scheduler.schedule();
        })
    });
}

fn benchmark_lpu_task(c: &mut Criterion) {
    let mut scheduler = create_test_scheduler();
    c.bench_function("LPU task", |b| {
        b.iter(|| {
            let task = create_test_task(black_box(1), ProcessingUnitType::LPU);
            scheduler.add_task(task);
            let _ = scheduler.schedule();
        })
    });
}

fn benchmark_npu_task(c: &mut Criterion) {
    let mut scheduler = create_test_scheduler();
    c.bench_function("NPU task", |b| {
        b.iter(|| {
            let task = create_test_task(black_box(1), ProcessingUnitType::NPU);
            scheduler.add_task(task);
            let _ = scheduler.schedule();
        })
    });
}

fn benchmark_fpga_task(c: &mut Criterion) {
    let mut scheduler = create_test_scheduler();
    c.bench_function("FPGA task", |b| {
        b.iter(|| {
            let task = create_test_task(black_box(1), ProcessingUnitType::FPGA);
            scheduler.add_task(task);
            let _ = scheduler.schedule();
        })
    });
}

fn benchmark_vpu_task(c: &mut Criterion) {
    let mut scheduler = create_test_scheduler();
    c.bench_function("VPU task", |b| {
        b.iter(|| {
            let task = create_test_task(black_box(1), ProcessingUnitType::VPU);
            scheduler.add_task(task);
            let _ = scheduler.schedule();
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
    benchmark_npu_task,
    benchmark_fpga_task,
    benchmark_vpu_task
);
criterion_main!(benches);
