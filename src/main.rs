mod xpu_optimization;

fn main() {
    println!("XPU Optimization Project");

    // Initialize XPU optimization
    let mut xpu_optimizer = xpu_optimization::XpuOptimizer::new();

    // Run optimization
    match xpu_optimizer.run() {
        Ok(_) => println!("XPU optimization completed successfully."),
        Err(e) => eprintln!("Error during XPU optimization: {}", e),
    }
}
