use std::collections::VecDeque;

pub struct MemoryManager {
    memory_pool: Vec<usize>,
    total_memory: usize,
    allocated_memory: usize,
}

impl MemoryManager {
    pub fn new(total_memory: usize) -> Self {
        MemoryManager {
            memory_pool: Vec::new(),
            total_memory,
            allocated_memory: 0,
        }
    }

    pub fn allocate(&mut self, size: usize) -> Result<(), String> {
        if self.allocated_memory + size > self.total_memory {
            return Err("Not enough memory available".to_string());
        }
        self.memory_pool.push(size);
        self.allocated_memory += size;
        Ok(())
    }

    // This method is currently unused but may be useful for future memory management features.
    // Consider implementing memory deallocation in task completion or error handling scenarios.
    pub fn deallocate(&mut self, size: usize) -> Result<(), String> {
        if let Some(index) = self.memory_pool.iter().position(|&x| x == size) {
            self.memory_pool.remove(index);
            self.allocated_memory -= size;
            Ok(())
        } else {
            Err("Memory block not found".to_string())
        }
    }

    pub fn get_available_memory(&self) -> usize {
        self.total_memory - self.allocated_memory
    }

    pub fn allocate_for_tasks(&mut self, tasks: &VecDeque<crate::task_scheduling::Task>) -> Result<(), String> {
        let total_required = tasks.iter().map(|task| task.memory_requirement).sum::<usize>();
        if self.allocated_memory + total_required <= self.total_memory {
            for task in tasks {
                self.allocate(task.memory_requirement)?;
            }
            Ok(())
        } else {
            Err("Not enough memory for all tasks".to_string())
        }
    }
}
