use crate::task_scheduling::{ProcessingUnit, ProcessingUnitType, Task};
use crate::XpuOptimizerError;
use std::time::Duration;

pub struct FPGACore {
    processing_unit: ProcessingUnit,
}

impl FPGACore {
    pub fn new(id: usize, processing_power: f32) -> Self {
        FPGACore {
            processing_unit: ProcessingUnit {
                id,
                unit_type: ProcessingUnitType::FPGA,
                current_load: Duration::new(0, 0),
                processing_power,
                power_state: crate::power_management::PowerState::Normal,
                energy_profile: crate::power_management::EnergyProfile::default(),
            },
        }
    }

    pub fn execute_task(&mut self, task: &Task) -> Result<Duration, XpuOptimizerError> {
        // Simulate task execution on FPGA
        let execution_time = task.execution_time.mul_f32(1.0 / self.processing_unit.processing_power);
        self.processing_unit.current_load += execution_time;

        // In a real implementation, we would program the FPGA and execute the task here
        println!("Executing task {} on FPGA {}", task.id, self.processing_unit.id);

        Ok(execution_time)
    }

    pub fn get_current_load(&self) -> Duration {
        self.processing_unit.current_load
    }

    pub fn reset_load(&mut self) {
        self.processing_unit.current_load = Duration::new(0, 0);
    }
}
