use crate::task_scheduling::{ProcessingUnit, ProcessingUnitType, Task};
use crate::power_management::{PowerState, EnergyProfile};
use std::time::Duration;

pub struct GPU {
    processing_unit: ProcessingUnit,
}

impl GPU {
    pub fn new(id: usize, processing_power: f32) -> Self {
        GPU {
            processing_unit: ProcessingUnit {
                id,
                unit_type: ProcessingUnitType::GPU,
                current_load: Duration::new(0, 0),
                processing_power,
                power_state: PowerState::Normal,
                energy_profile: EnergyProfile::default(),
            },
        }
    }

    pub fn execute_task(&mut self, task: &Task) -> Duration {
        // Simulate task execution
        let execution_time = task.execution_time;
        self.processing_unit.current_load += execution_time;

        // Adjust power state based on current load
        self.adjust_power_state();

        execution_time
    }

    fn adjust_power_state(&mut self) {
        let load_percentage = self.processing_unit.current_load.as_secs_f32() / self.processing_unit.processing_power;
        self.processing_unit.power_state = match load_percentage {
            x if x < 0.3 => PowerState::LowPower,
            x if x < 0.7 => PowerState::Normal,
            _ => PowerState::HighPerformance,
        };
    }

    pub fn get_current_load(&self) -> Duration {
        self.processing_unit.current_load
    }

    pub fn get_power_state(&self) -> &PowerState {
        &self.processing_unit.power_state
    }
}
