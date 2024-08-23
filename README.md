# XPU Manager Rust
[![XPU CI/CD Pipeline](https://github.com/VishwamAI/xpu_1/actions/workflows/xpu.yaml/badge.svg)](https://github.com/VishwamAI/xpu_1/actions/workflows/xpu.yaml)

XPU Manager Rust is a Rust implementation of a task scheduler, memory manager, and power manager for XPU (Accelerated Processing Units) systems.

## Project Overview

This project provides a basic framework for managing tasks, memory, and power in an XPU environment. It includes:

- Task Scheduling: Prioritize and schedule tasks across multiple processing units.
- Memory Management: Allocate and manage memory for tasks.
- Power Management: Optimize power consumption based on system load.
- Cloud Offloading: Efficiently manage workload distribution between local and cloud resources.
- Adaptive Optimization: Implement ML-driven strategies for system optimization.
- Inspired by Intel XPU development benchmarks and development

  ![image](https://github.com/user-attachments/assets/3a636d17-d4e4-4fb8-a7b7-2ed70257ab33)

## Recent Updates

- Improved power management policy handling with case-insensitive input.
- Fixed `test_configure_xpu_manager` test failure related to power management policy configuration.
- Enhanced error handling and logging throughout the codebase.
- Implemented cloud offloading and adaptive optimization features.

## Installation

To use XPU Manager Rust, you need to have Rust installed on your system. If you don't have Rust installed, you can get it from [https://www.rust-lang.org/](https://www.rust-lang.org/).

1. Clone the repository:
   ```
   git clone https://github.com/VishwamAI/xpu_1.git
   cd xpu_1
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
use xpu_manager_rust::{XpuOptimizer, XpuOptimizerConfig, Task};
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = XpuOptimizerConfig::default();
    let mut optimizer = XpuOptimizer::new(config)?;

    let task = Task {
        id: 1,
        priority: 2,
        execution_time: Duration::from_secs(5),
        memory_requirement: 200,
    };

    optimizer.add_task(task)?;
    optimizer.run()?;

    Ok(())
}
```

## Benchmarks

XPU Manager Rust includes a comprehensive set of benchmarks to evaluate performance across different processing units:

1. CPU Benchmark: Measures task scheduling and execution performance on CPUs.
2. GPU Benchmark: Evaluates parallel processing capabilities using GPUs.
3. TPU Benchmark: Tests tensor processing operations on TPUs.
4. NPU Benchmark: Assesses neural network inference performance on NPUs.
5. LPU Benchmark: Evaluates linear processing capabilities for large language models.

These benchmarks are conducted using Criterion.rs, a statistics-driven micro-benchmarking tool. Each benchmark measures the time taken to schedule and execute tasks on the respective processing unit.

To run the benchmarks locally:

```
cargo bench
```

## Performance Benchmarks

Recent benchmark results:

| Processor | Task                                 | Time (µs) | Variation |
|-----------|--------------------------------------|-----------|-----------|
| CPU       | Task Scheduling                      | 10.611    | ±22.1%    |
| GPU       | Parallel Task Execution              | 15.234    | ±18.7%    |
| TPU       | Tensor Operations                    | 8.756     | ±15.3%    |
| NPU       | Neural Network Inference             | 12.089    | ±20.5%    |
| LPU       | Linear Processing for LLMs           | 7.123     | ±12.8%    |

Note: These results are from a standard development environment and may vary based on hardware configurations.

Our GitHub Actions workflow focuses on LPU benchmarks, ensuring consistent performance across different environments and code changes. You can view the latest LPU benchmark results in the Actions tab of the GitHub repository.

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
