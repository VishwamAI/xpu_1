use serde::{Deserialize, Serialize};

pub struct PowerManager {
    current_power_state: PowerState,
    power_consumption: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PowerState {
    LowPower,
    Normal,
    HighPerformance,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnergyProfile {
    pub consumption_rate: f32,
}

impl Default for EnergyProfile {
    fn default() -> Self {
        EnergyProfile {
            consumption_rate: 1.0, // Default consumption rate
        }
    }
}

impl PowerManager {
    pub fn new() -> Self {
        PowerManager {
            current_power_state: PowerState::Normal,
            power_consumption: 0.0,
        }
    }

    pub fn set_power_state(&mut self, state: PowerState) {
        self.current_power_state = state;
        // Update power consumption based on the new state
        self.update_power_consumption();
    }

    pub fn get_power_state(&self) -> &PowerState {
        &self.current_power_state
    }

    pub fn get_power_consumption(&self) -> f32 {
        self.power_consumption
    }

    fn update_power_consumption(&mut self) {
        // Simulate power consumption based on the current state
        self.power_consumption = match self.current_power_state {
            PowerState::LowPower => 0.5,
            PowerState::Normal => 1.0,
            PowerState::HighPerformance => 2.0,
        };
    }

    pub fn optimize_power(&mut self, load: f32) {
        // Simple power optimization logic
        if load < 0.3 {
            self.set_power_state(PowerState::LowPower);
        } else if load < 0.7 {
            self.set_power_state(PowerState::Normal);
        } else {
            self.set_power_state(PowerState::HighPerformance);
        }
    }
}

impl Default for PowerManager {
    fn default() -> Self {
        Self::new()
    }
}
