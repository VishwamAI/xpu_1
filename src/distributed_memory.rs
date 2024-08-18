use std::collections::HashMap;
use crate::XpuOptimizerError;

pub trait DistributedMemoryManager: Send + Sync {
    fn allocate(&mut self, task_id: usize, size: usize) -> Result<(), XpuOptimizerError>;
    fn deallocate(&mut self, task_id: usize) -> Result<(), XpuOptimizerError>;
    fn get_memory_usage(&self) -> usize;
}

pub struct SimpleDistributedMemoryManager {
    memory_pool: HashMap<usize, usize>,
    total_memory: usize,
    used_memory: usize,
}

impl SimpleDistributedMemoryManager {
    pub fn new(total_memory: usize) -> Self {
        SimpleDistributedMemoryManager {
            memory_pool: HashMap::new(),
            total_memory,
            used_memory: 0,
        }
    }
}

impl DistributedMemoryManager for SimpleDistributedMemoryManager {
    fn allocate(&mut self, task_id: usize, size: usize) -> Result<(), XpuOptimizerError> {
        if self.used_memory + size > self.total_memory {
            return Err(XpuOptimizerError::MemoryError("Not enough memory available".to_string()));
        }
        self.memory_pool.insert(task_id, size);
        self.used_memory += size;
        Ok(())
    }

    fn deallocate(&mut self, task_id: usize) -> Result<(), XpuOptimizerError> {
        if let Some(size) = self.memory_pool.remove(&task_id) {
            self.used_memory -= size;
            Ok(())
        } else {
            Err(XpuOptimizerError::MemoryError("Task not found in memory pool".to_string()))
        }
    }

    fn get_memory_usage(&self) -> usize {
        self.used_memory
    }
}
