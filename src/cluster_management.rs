use crate::{ProcessingUnit, Task, XpuOptimizerError, PowerState, EnergyProfile, ProcessingUnitType};
use std::collections::HashMap;

pub trait ClusterManager: Send + Sync {
    fn add_node(&mut self, node: ClusterNode) -> Result<(), XpuOptimizerError>;
    fn remove_node(&mut self, node_id: &str) -> Result<(), XpuOptimizerError>;
    fn get_node(&self, node_id: &str) -> Option<&ClusterNode>;
    fn list_nodes(&self) -> Vec<&ClusterNode>;
    fn update_node_status(
        &mut self,
        node_id: &str,
        status: NodeStatus,
    ) -> Result<(), XpuOptimizerError>;
}

pub trait LoadBalancer: Send + Sync {
    fn distribute_tasks(
        &self,
        tasks: &[Task],
        nodes: &[ClusterNode],
    ) -> Result<HashMap<String, Vec<Task>>, XpuOptimizerError>;
}

#[derive(Debug, Clone)]
pub struct ClusterNode {
    pub id: String,
    pub ip_address: String,
    pub processing_units: Vec<ProcessingUnit>,
    pub status: NodeStatus,
    pub current_load: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeStatus {
    Active,
    Inactive,
    Maintenance,
}

// Implement a basic cluster manager
#[derive(Default)]
pub struct SimpleClusterManager {
    nodes: HashMap<String, ClusterNode>,
}

impl SimpleClusterManager {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ClusterManager for SimpleClusterManager {
    fn add_node(&mut self, node: ClusterNode) -> Result<(), XpuOptimizerError> {
        if self.nodes.contains_key(&node.id) {
            return Err(XpuOptimizerError::ClusterInitializationError(format!(
                "Node with ID {} already exists",
                node.id
            )));
        }
        self.nodes.insert(node.id.clone(), node);
        Ok(())
    }

    fn remove_node(&mut self, node_id: &str) -> Result<(), XpuOptimizerError> {
        if self.nodes.remove(node_id).is_none() {
            return Err(XpuOptimizerError::ClusterInitializationError(format!(
                "Node with ID {} not found",
                node_id
            )));
        }
        Ok(())
    }

    fn get_node(&self, node_id: &str) -> Option<&ClusterNode> {
        self.nodes.get(node_id)
    }

    fn list_nodes(&self) -> Vec<&ClusterNode> {
        self.nodes.values().collect()
    }

    fn update_node_status(
        &mut self,
        node_id: &str,
        status: NodeStatus,
    ) -> Result<(), XpuOptimizerError> {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.status = status;
            Ok(())
        } else {
            Err(XpuOptimizerError::ClusterInitializationError(format!(
                "Node with ID {} not found",
                node_id
            )))
        }
    }
}

// Implement a basic load balancer
#[derive(Default)]
pub struct RoundRobinLoadBalancer;

impl LoadBalancer for RoundRobinLoadBalancer {
    fn distribute_tasks(
        &self,
        tasks: &[Task],
        nodes: &[ClusterNode],
    ) -> Result<HashMap<String, Vec<Task>>, XpuOptimizerError> {
        let mut distribution: HashMap<String, Vec<Task>> = HashMap::new();
        let active_nodes: Vec<_> = nodes
            .iter()
            .filter(|n| n.status == NodeStatus::Active)
            .collect();

        if active_nodes.is_empty() {
            return Err(XpuOptimizerError::ClusterInitializationError(
                "No active nodes available".to_string(),
            ));
        }

        for (i, task) in tasks.iter().enumerate() {
            let node = &active_nodes[i % active_nodes.len()];
            distribution
                .entry(node.id.clone())
                .or_default()
                .push(task.clone());
        }

        Ok(distribution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_node(id: &str) -> ClusterNode {
        ClusterNode {
            id: id.to_string(),
            ip_address: format!("192.168.1.{}", id),
            processing_units: vec![ProcessingUnit {
                id: 0,
                unit_type: ProcessingUnitType::CPU,
                current_load: std::time::Duration::new(0, 0),
                processing_power: 1.0,
                power_state: PowerState::Normal,
                energy_profile: EnergyProfile::default(),
            }],
            status: NodeStatus::Active,
            current_load: 0.0,
        }
    }

    #[test]
    fn test_simple_cluster_manager() {
        let mut manager = SimpleClusterManager::new();

        // Test add_node
        let node1 = create_test_node("1");
        assert!(manager.add_node(node1.clone()).is_ok());
        assert!(manager.add_node(node1.clone()).is_err());

        // Test get_node
        assert!(manager.get_node("1").is_some());
        assert!(manager.get_node("2").is_none());

        // Test update_node_status
        assert!(manager.update_node_status("1", NodeStatus::Maintenance).is_ok());
        assert_eq!(manager.get_node("1").unwrap().status, NodeStatus::Maintenance);

        // Test remove_node
        assert!(manager.remove_node("1").is_ok());
        assert!(manager.remove_node("1").is_err());
    }

    #[test]
    fn test_round_robin_load_balancer() {
        let balancer = RoundRobinLoadBalancer;
        let nodes = vec![
            create_test_node("1"),
            create_test_node("2"),
            create_test_node("3"),
        ];
        let tasks = vec![
            Task { id: 1, priority: 1, execution_time: std::time::Duration::new(1, 0), memory_requirement: 1024, unit_type: ProcessingUnitType::CPU, unit: ProcessingUnit { id: 0, unit_type: ProcessingUnitType::CPU, current_load: std::time::Duration::new(0, 0), processing_power: 1.0, power_state: PowerState::Normal, energy_profile: EnergyProfile::default() }, dependencies: vec![], secure: false, estimated_duration: std::time::Duration::new(1, 0), estimated_resource_usage: 1024 },
            Task { id: 2, priority: 1, execution_time: std::time::Duration::new(1, 0), memory_requirement: 1024, unit_type: ProcessingUnitType::CPU, unit: ProcessingUnit { id: 0, unit_type: ProcessingUnitType::CPU, current_load: std::time::Duration::new(0, 0), processing_power: 1.0, power_state: PowerState::Normal, energy_profile: EnergyProfile::default() }, dependencies: vec![], secure: false, estimated_duration: std::time::Duration::new(1, 0), estimated_resource_usage: 1024 },
            Task { id: 3, priority: 1, execution_time: std::time::Duration::new(1, 0), memory_requirement: 1024, unit_type: ProcessingUnitType::CPU, unit: ProcessingUnit { id: 0, unit_type: ProcessingUnitType::CPU, current_load: std::time::Duration::new(0, 0), processing_power: 1.0, power_state: PowerState::Normal, energy_profile: EnergyProfile::default() }, dependencies: vec![], secure: false, estimated_duration: std::time::Duration::new(1, 0), estimated_resource_usage: 1024 },
            Task { id: 4, priority: 1, execution_time: std::time::Duration::new(1, 0), memory_requirement: 1024, unit_type: ProcessingUnitType::CPU, unit: ProcessingUnit { id: 0, unit_type: ProcessingUnitType::CPU, current_load: std::time::Duration::new(0, 0), processing_power: 1.0, power_state: PowerState::Normal, energy_profile: EnergyProfile::default() }, dependencies: vec![], secure: false, estimated_duration: std::time::Duration::new(1, 0), estimated_resource_usage: 1024 },
        ];

        let distribution = balancer.distribute_tasks(&tasks, &nodes).unwrap();
        assert_eq!(distribution.len(), 3);
        assert_eq!(distribution.get("1").unwrap().len(), 2);
        assert_eq!(distribution.get("2").unwrap().len(), 1);
        assert_eq!(distribution.get("3").unwrap().len(), 1);
    }
}
