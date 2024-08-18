use std::time::Duration;

pub trait ScalingPolicy: Send + Sync {
    fn determine_scaling_action(
        &self,
        current_load: f32,
        available_resources: usize,
    ) -> ScalingAction;
}

pub enum ScalingAction {
    ScaleUp(usize),
    ScaleDown(usize),
    NoAction,
}

pub struct DynamicScalingPolicy {
    scale_up_threshold: f32,
    scale_down_threshold: f32,
    cooldown_period: Duration,
    last_action_time: Option<std::time::Instant>,
}

impl DynamicScalingPolicy {
    pub fn new(
        scale_up_threshold: f32,
        scale_down_threshold: f32,
        cooldown_period: Duration,
    ) -> Self {
        Self {
            scale_up_threshold,
            scale_down_threshold,
            cooldown_period,
            last_action_time: None,
        }
    }
}

impl ScalingPolicy for DynamicScalingPolicy {
    fn determine_scaling_action(
        &self,
        current_load: f32,
        available_resources: usize,
    ) -> ScalingAction {
        let now = std::time::Instant::now();

        if let Some(last_action) = self.last_action_time {
            if now.duration_since(last_action) < self.cooldown_period {
                return ScalingAction::NoAction;
            }
        }

        if current_load > self.scale_up_threshold {
            ScalingAction::ScaleUp(1)
        } else if current_load < self.scale_down_threshold && available_resources > 1 {
            ScalingAction::ScaleDown(1)
        } else {
            ScalingAction::NoAction
        }
    }
}
