use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Not enough memory available")]
    InsufficientMemory,
    #[error("Memory block not found")]
    BlockNotFound,
    #[error("Memory block size not found")]
    SizeNotFound,
}

pub struct MemoryManager {
    memory_pool: std::collections::BTreeMap<usize, Vec<usize>>,
    total_memory: usize,
    allocated_memory: usize,
    fragmentation_threshold: f32,
}

impl MemoryManager {
    pub fn new(total_memory: usize) -> Self {
        MemoryManager {
            memory_pool: BTreeMap::new(),
            total_memory,
            allocated_memory: 0,
            fragmentation_threshold: 0.2, // 20% fragmentation threshold
        }
    }

    pub fn allocate(&mut self, size: usize) -> Result<(), String> {
        if self.allocated_memory + size > self.total_memory {
            return Err("Not enough memory available".to_string());
        }

        let mut allocated = false;
        for (block_size, blocks) in self.memory_pool.range_mut(size..) {
            if let Some(block) = blocks.pop() {
                self.allocated_memory += size;
                if *block_size > size {
                    let remaining = *block_size - size;
                    self.memory_pool.entry(remaining).or_default().push(block + size);
                }
                allocated = true;
                break;
            }
        }

        if !allocated {
            self.memory_pool.entry(size).or_default().push(self.allocated_memory);
            self.allocated_memory += size;
        }

        if self.fragmentation_level() > self.fragmentation_threshold {
            self.defragment();
        }

        Ok(())
    }

    pub fn deallocate(&mut self, size: usize) -> Result<(), String> {
        if let Some(blocks) = self.memory_pool.get_mut(&size) {
            if let Some(_) = blocks.pop() {
                self.allocated_memory = self.allocated_memory.saturating_sub(size);
                Ok(())
            } else {
                Err("Memory block not found".to_string())
            }
        } else {
            Err("Memory block size not found".to_string())
        }
    }

    pub fn get_available_memory(&self) -> usize {
        self.total_memory - self.allocated_memory
    }

    pub fn allocate_for_tasks(
        &mut self,
        tasks: &[crate::task_scheduling::Task],
    ) -> Result<(), String> {
        let total_required: usize = tasks.iter().map(|task| task.memory_requirement).sum();
        let available_memory = self.get_available_memory();

        if total_required > available_memory {
            return Err(format!(
                "Not enough memory for all tasks. Available: {} bytes, Required: {} bytes",
                available_memory,
                total_required
            ));
        }

        for task in tasks {
            self.allocate(task.memory_requirement)?;
        }

        Ok(())
    }

    pub fn deallocate_completed_tasks(
        &mut self,
        completed_tasks: &[crate::task_scheduling::Task],
    ) -> Result<(), String> {
        for task in completed_tasks {
            self.deallocate(task.memory_requirement)?;
        }
        Ok(())
    }

    fn fragmentation_level(&self) -> f32 {
        let free_blocks: usize = self.memory_pool.values().map(|blocks| blocks.len()).sum();
        free_blocks as f32 / self.total_memory as f32
    }

    fn defragment(&mut self) {
        let mut new_pool: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut current_address = 0;

        for (size, blocks) in self.memory_pool.iter() {
            for _ in blocks {
                new_pool.entry(*size).or_default().push(current_address);
                current_address += size;
            }
        }

        self.memory_pool = new_pool;
    }

    pub fn get_fragmentation_percentage(&self) -> f32 {
        self.fragmentation_level() * 100.0
    }
}
