use crate::task_scheduling::{ProcessingUnit, ProcessingUnitType, Task};
use crate::power_management::{PowerState, EnergyProfile};
use std::time::Duration;

pub struct TPU {
    id: usize,
    processing_power: f32,
    current_load: Duration,
    power_state: PowerState,
    energy_profile: EnergyProfile,
}

impl TPU {
    pub fn new(id: usize, processing_power: f32) -> Self {
        TPU {
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
            Err("TPU cannot handle this task".to_string())
        }
    }

    fn can_handle_task(&self, task: &Task) -> bool {
        task.unit_type == ProcessingUnitType::TPU
    }

    fn calculate_execution_time(&self, task: &Task) -> Duration {
        let base_time = task.execution_time;
        let adjusted_time = base_time.mul_f32(1.0 / self.processing_power);
        adjusted_time
    }

    fn adjust_power_state(&mut self) {
        self.power_state = match self.current_load.as_secs_f32() / self.processing_power {
            load if load < 0.3 => PowerState::LowPower,
            load if load < 0.7 => PowerState::Normal,
            _ => PowerState::HighPerformance,
        };
    }
}

impl ProcessingUnit for TPU {
    fn id(&self) -> usize {
        self.id
    }

    fn unit_type(&self) -> ProcessingUnitType {
        ProcessingUnitType::TPU
    }

    fn current_load(&self) -> Duration {
        self.current_load
    }

    fn processing_power(&self) -> f32 {
        self.processing_power
    }

    fn power_state(&self) -> &PowerState {
        &self.power_state
    }

    fn energy_profile(&self) -> &EnergyProfile {
        &self.energy_profile
    }
}
