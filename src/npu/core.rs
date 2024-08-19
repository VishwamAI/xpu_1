use crate::task_scheduling::{ProcessingUnit, ProcessingUnitType, Task};
use crate::power_management::{PowerState, EnergyProfile};
use std::time::Duration;

pub struct NPU {
    processing_unit: ProcessingUnit,
}

impl NPU {
    pub fn new(id: usize) -> Self {
        NPU {
            processing_unit: ProcessingUnit {
                id,
                unit_type: ProcessingUnitType::NPU,
                current_load: Duration::new(0, 0),
                processing_power: 1.0,
                power_state: PowerState::Normal,
                energy_profile: EnergyProfile::default(),
            },
        }
    }

    pub fn process_task(&mut self, task: &Task) -> Result<Duration, String> {
        if self.processing_unit.unit_type != task.unit.unit_type {
            return Err("Task is not compatible with NPU".to_string());
        }

        let processing_time = task.execution_time.div_f32(self.processing_unit.processing_power);
        self.processing_unit.current_load += processing_time;

        Ok(processing_time)
    }

    pub fn get_current_load(&self) -> Duration {
        self.processing_unit.current_load
    }

    pub fn set_power_state(&mut self, state: PowerState) {
        self.processing_unit.power_state = state;
    }

    pub fn get_power_state(&self) -> &PowerState {
        &self.processing_unit.power_state
    }

    pub fn get_energy_profile(&self) -> &EnergyProfile {
        &self.processing_unit.energy_profile
    }
}
