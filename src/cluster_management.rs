use crate::{ProcessingUnit, Task, XpuOptimizerError};
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
