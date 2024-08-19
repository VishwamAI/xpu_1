# XPU Manager Rust
[![XPU CI/CD Pipeline](https://github.com/VishwamAI/xpu_1/actions/workflows/xpu.yaml/badge.svg)](https://github.com/VishwamAI/xpu_1/actions/workflows/xpu.yaml)

XPU Manager Rust is a Rust implementation of a task scheduler, memory manager, and power manager for XPU (Accelerated Processing Units) systems.

## Project Overview

This project provides a basic framework for managing tasks, memory, and power in an XPU environment. It includes:

- Task Scheduling: Prioritize and schedule tasks across multiple processing units.
- Memory Management: Allocate and manage memory for tasks.
- Power Management: Optimize power consumption based on system load.
- Inspired by intel xpu devlopment benchmarks and devlopment

  ![image](https://github.com/user-attachments/assets/3a636d17-d4e4-4fb8-a7b7-2ed70257ab33)


## Installation

To use XPU Manager Rust, you need to have Rust installed on your system. If you don't have Rust installed, you can get it from [https://www.rust-lang.org/](https://www.rust-lang.org/).

1. Clone the repository:
   ```
   git clone https://https://github.com/VishwamAI/xpu_1.git
   cd xpu_manager_rust
   ```

2. Build the project:
   ```
   cargo build
   ```

## Usage

To run the XPU Manager Rust demo:

```
cargo run
```

This will execute a sample scenario demonstrating task scheduling, memory allocation, and power management.

## Usage Examples

Here's a basic example of how to use the XPU Manager Rust components:

```rust
use xpu_manager_rust::{TaskScheduler, Task, MemoryManager, PowerManager};
use std::time::Duration;

fn main() {
    let mut scheduler = TaskScheduler::new(4);
    let mut memory_manager = MemoryManager::new(1024);
    let mut power_manager = PowerManager::new();

    let task = Task {
        id: 1,
        priority: 2,
        execution_time: Duration::from_secs(5),
        memory_requirement: 200,
    };

    scheduler.add_task(task);
    memory_manager.allocate_for_tasks(&scheduler.tasks).unwrap();
    scheduler.schedule();

    power_manager.optimize_power(0.6);
}
```

## Benchmarks

XPU Manager Rust includes a set of benchmarks to evaluate performance across different processing units:

1. CPU Benchmark: Measures task scheduling and execution performance on CPUs.
2. GPU Benchmark: Evaluates parallel processing capabilities using GPUs.
3. TPU Benchmark: Tests tensor processing operations on TPUs.
4. NPU Benchmark: Assesses neural network inference performance on NPUs.

These benchmarks are integrated into our GitHub Actions workflow, ensuring consistent performance across different environments and code changes. To run the benchmarks locally:

```
cargo bench
```

The GitHub Actions workflow automatically runs these benchmarks on every push and pull request to the main branch. You can view the benchmark results in the Actions tab of the GitHub repository.

## Contributing

Contributions to XPU Manager Rust are welcome! Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

Please ensure your code adheres to the existing style and passes all tests before submitting a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
