use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PowerManagementError {
    #[error("Invalid power state transition: {0}")]
    InvalidStateTransition(String),
    #[error("Energy consumption calculation error: {0}")]
    EnergyCalculationError(String),
}

pub struct PowerManager {
    current_power_state: PowerState,
    power_consumption: f64,
    energy_monitor: EnergyMonitor,
    power_policy: PowerPolicy,
}

impl PowerManager {
    pub fn set_policy(&mut self, policy: PowerManagementPolicy) {
        self.power_policy = match policy {
            PowerManagementPolicy::Default => PowerPolicy::default(),
            PowerManagementPolicy::Aggressive => PowerPolicy {
                low_power_threshold: 0.2,
                high_power_threshold: 0.6,
            },
            PowerManagementPolicy::Conservative => PowerPolicy {
                low_power_threshold: 0.4,
                high_power_threshold: 0.8,
            },
        };
    }
}

impl PowerManager {
    pub fn new() -> Self {
        PowerManager {
            current_power_state: PowerState::Normal,
            power_consumption: 0.0,
            energy_monitor: EnergyMonitor::new(),
            power_policy: PowerPolicy::default(),
        }
    }

    pub fn set_power_state(&mut self, state: PowerState) -> Result<(), PowerManagementError> {
        self.current_power_state = state;
        self.update_power_consumption()
    }

    pub fn get_power_state(&self) -> &PowerState {
        &self.current_power_state
    }

    pub fn get_power_consumption(&self) -> f64 {
        self.power_consumption
    }

    fn update_power_consumption(&mut self) -> Result<(), PowerManagementError> {
        self.power_consumption = match self.current_power_state {
            PowerState::LowPower => 0.5,
            PowerState::Normal => 1.0,
            PowerState::HighPerformance => 2.0,
        };
        self.energy_monitor.record_energy_consumption(self.power_consumption)
            .map_err(|e| PowerManagementError::EnergyCalculationError(e.to_string()))
    }

    pub fn optimize_power(&mut self, load: f64) -> Result<(), PowerManagementError> {
        let new_state = if load < self.power_policy.low_power_threshold {
            PowerState::LowPower
        } else if load < self.power_policy.high_power_threshold {
            PowerState::Normal
        } else {
            PowerState::HighPerformance
        };
        self.set_power_state(new_state)
    }

    pub fn set_power_policy(&mut self, policy: PowerPolicy) {
        self.power_policy = policy;
    }

    pub fn get_total_energy_consumed(&self) -> Result<f64, PowerManagementError> {
        Ok(self.energy_monitor.get_total_energy_consumed())
    }
}

impl Default for PowerManager {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PowerState {
    LowPower,
    Normal,
    HighPerformance,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnergyProfile {
    pub consumption_rate: f64,
}

impl Default for EnergyProfile {
    fn default() -> Self {
        EnergyProfile {
            consumption_rate: 1.0, // Default consumption rate
        }
    }
}

pub struct EnergyMonitor {
    total_energy_consumed: f64,
}

impl EnergyMonitor {
    pub fn new() -> Self {
        EnergyMonitor {
            total_energy_consumed: 0.0,
        }
    }

    pub fn record_energy_consumption(&mut self, energy: f64) -> Result<(), PowerManagementError> {
        self.total_energy_consumed += energy;
        if self.total_energy_consumed.is_finite() {
            Ok(())
        } else {
            Err(PowerManagementError::EnergyCalculationError("Energy consumption overflow".to_string()))
        }
    }

    pub fn get_total_energy_consumed(&self) -> f64 {
        self.total_energy_consumed
    }
}

pub struct PowerPolicy {
    low_power_threshold: f64,
    high_power_threshold: f64,
}

impl Default for PowerPolicy {
    fn default() -> Self {
        PowerPolicy {
            low_power_threshold: 0.3,
            high_power_threshold: 0.7,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PowerManagementPolicy {
    Default,
    Aggressive,
    Conservative,
}
