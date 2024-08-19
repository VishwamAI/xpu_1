use crate::task_scheduling::{ProcessingUnit, ProcessingUnitType, Task};
use crate::power_management::{PowerState, EnergyProfile};
use std::time::Duration;

pub struct LPU {
    id: usize,
    processing_power: f32,
    current_load: Duration,
    power_state: PowerState,
    energy_profile: EnergyProfile,
}

impl LPU {
    pub fn new(id: usize, processing_power: f32) -> Self {
        LPU {
            id,
            processing_power,
            current_load: Duration::new(0, 0),
            power_state: PowerState::Normal,
            energy_profile: EnergyProfile::default(),
        }
    }

    pub fn process_task(&mut self, task: &Task) -> Result<Duration, String> {
        if self.can_handle_task(task) {
            let execution_time = self.calculate_execution_time(task);
            self.current_load += execution_time;
            self.adjust_power_state();
            Ok(execution_time)
        } else {
            Err("LPU cannot handle this task".to_string())
        }
    }

    fn can_handle_task(&self, task: &Task) -> bool {
        task.unit_type == ProcessingUnitType::LPU && self.available_capacity() >= task.execution_time
    }

    fn available_capacity(&self) -> Duration {
        Duration::from_secs_f32(self.processing_power) - self.current_load
    }

    fn calculate_execution_time(&self, task: &Task) -> Duration {
        // LPU-specific execution time calculation
        let base_time = task.execution_time;
        match self.power_state {
            PowerState::LowPower => base_time.mul_f32(1.5),
            PowerState::Normal => base_time,
            PowerState::HighPerformance => base_time.mul_f32(0.8),
        }
    }

    fn adjust_power_state(&mut self) {
        let load_percentage = self.current_load.as_secs_f32() / self.processing_power;
        self.power_state = match load_percentage {
            x if x < 0.3 => PowerState::LowPower,
            x if x < 0.7 => PowerState::Normal,
            _ => PowerState::HighPerformance,
        };
    }

    pub fn get_current_power_consumption(&self) -> f32 {
        self.energy_profile.consumption_rate * match self.power_state {
            PowerState::LowPower => 0.5,
            PowerState::Normal => 1.0,
            PowerState::HighPerformance => 1.5,
        }
    }
}

impl From<LPU> for ProcessingUnit {
    fn from(lpu: LPU) -> Self {
        ProcessingUnit {
            id: lpu.id,
            unit_type: ProcessingUnitType::LPU,
            current_load: lpu.current_load,
            processing_power: lpu.processing_power,
            power_state: lpu.power_state,
            energy_profile: lpu.energy_profile,
        }
    }
}
