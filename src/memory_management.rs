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

pub trait MemoryManager {
    fn allocate(&mut self, size: usize) -> Result<(), MemoryError>;
    fn deallocate(&mut self, size: usize) -> Result<(), MemoryError>;
    fn get_available_memory(&self) -> usize;
    fn allocate_for_tasks(&mut self, tasks: &[crate::task_scheduling::Task]) -> Result<(), MemoryError>;
    fn deallocate_completed_tasks(&mut self, completed_tasks: &[crate::task_scheduling::Task]) -> Result<(), MemoryError>;
    fn force_free(&mut self, size: usize) -> Result<(), MemoryError>;
}

#[derive(Debug, Clone, Copy)]
pub enum MemoryManagerType {
    Simple,
    Dynamic,
}

pub struct SimpleMemoryManager {
    memory_pool: BTreeMap<usize, Vec<usize>>,
    total_memory: usize,
    allocated_memory: usize,
    fragmentation_threshold: f32,
}

impl SimpleMemoryManager {
    pub fn new(total_memory: usize) -> Self {
        SimpleMemoryManager {
            memory_pool: BTreeMap::new(),
            total_memory,
            allocated_memory: 0,
            fragmentation_threshold: 0.2, // 20% fragmentation threshold
        }
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

impl MemoryManager for SimpleMemoryManager {
    fn allocate(&mut self, size: usize) -> Result<(), MemoryError> {
        if self.allocated_memory + size > self.total_memory {
            return Err(MemoryError::InsufficientMemory);
        }

        let mut allocated = false;
        for (block_size, blocks) in self.memory_pool.range_mut(size..) {
            if let Some(block) = blocks.pop() {
                self.allocated_memory += size;
                if *block_size > size {
                    let remaining = *block_size - size;
                    self.memory_pool
                        .entry(remaining)
                        .or_default()
                        .push(block + size);
                }
                allocated = true;
                break;
            }
        }

        if !allocated {
            self.memory_pool
                .entry(size)
                .or_default()
                .push(self.allocated_memory);
            self.allocated_memory += size;
        }

        if self.fragmentation_level() > self.fragmentation_threshold {
            self.defragment();
        }

        Ok(())
    }

    fn deallocate(&mut self, size: usize) -> Result<(), MemoryError> {
        if let Some(blocks) = self.memory_pool.get_mut(&size) {
            if blocks.pop().is_some() {
                self.allocated_memory = self.allocated_memory.saturating_sub(size);
                Ok(())
            } else {
                Err(MemoryError::BlockNotFound)
            }
        } else {
            Err(MemoryError::SizeNotFound)
        }
    }

    fn get_available_memory(&self) -> usize {
        self.total_memory - self.allocated_memory
    }

    fn allocate_for_tasks(&mut self, tasks: &[crate::task_scheduling::Task]) -> Result<(), MemoryError> {
        let total_required: usize = tasks.iter().map(|task| task.memory_requirement).sum();
        let available_memory = self.get_available_memory();

        if total_required > available_memory {
            return Err(MemoryError::InsufficientMemory);
        }

        for task in tasks {
            self.allocate(task.memory_requirement)?;
        }

        Ok(())
    }

    fn deallocate_completed_tasks(&mut self, completed_tasks: &[crate::task_scheduling::Task]) -> Result<(), MemoryError> {
        for task in completed_tasks {
            self.deallocate(task.memory_requirement)?;
        }
        Ok(())
    }

    fn force_free(&mut self, size: usize) -> Result<(), MemoryError> {
        if size > self.allocated_memory {
            return Err(MemoryError::InsufficientMemory);
        }

        let mut freed = 0;
        let mut to_remove = Vec::new();

        for (&block_size, blocks) in self.memory_pool.iter_mut() {
            while freed < size && !blocks.is_empty() {
                blocks.pop();
                freed += block_size;
            }
            if blocks.is_empty() {
                to_remove.push(block_size);
            }
            if freed >= size {
                break;
            }
        }

        for block_size in to_remove {
            self.memory_pool.remove(&block_size);
        }

        self.allocated_memory = self.allocated_memory.saturating_sub(freed);

        if freed < size {
            Err(MemoryError::InsufficientMemory)
        } else {
            Ok(())
        }
    }
}

pub struct DynamicMemoryManager {
    memory_pool: BTreeMap<usize, Vec<usize>>,
    total_memory: usize,
    allocated_memory: usize,
    block_size: usize,
}

impl DynamicMemoryManager {
    pub fn new(block_size: usize, total_memory: usize) -> Self {
        DynamicMemoryManager {
            memory_pool: BTreeMap::new(),
            total_memory,
            allocated_memory: 0,
            block_size,
        }
    }
}

impl MemoryManager for DynamicMemoryManager {
    fn allocate(&mut self, size: usize) -> Result<(), MemoryError> {
        let blocks_needed = (size + self.block_size - 1) / self.block_size;
        let total_size = blocks_needed * self.block_size;

        if self.allocated_memory + total_size > self.total_memory {
            return Err(MemoryError::InsufficientMemory);
        }

        let start_address = self.allocated_memory;
        self.memory_pool.entry(total_size).or_default().push(start_address);
        self.allocated_memory += total_size;

        Ok(())
    }

    fn deallocate(&mut self, size: usize) -> Result<(), MemoryError> {
        let blocks_needed = (size + self.block_size - 1) / self.block_size;
        let total_size = blocks_needed * self.block_size;

        if let Some(addresses) = self.memory_pool.get_mut(&total_size) {
            if let Some(_) = addresses.pop() {
                self.allocated_memory -= total_size;
                Ok(())
            } else {
                Err(MemoryError::BlockNotFound)
            }
        } else {
            Err(MemoryError::SizeNotFound)
        }
    }

    fn get_available_memory(&self) -> usize {
        self.total_memory - self.allocated_memory
    }

    fn allocate_for_tasks(&mut self, tasks: &[crate::task_scheduling::Task]) -> Result<(), MemoryError> {
        let total_required: usize = tasks.iter().map(|task| task.memory_requirement).sum();

        if total_required > self.get_available_memory() {
            return Err(MemoryError::InsufficientMemory);
        }

        for task in tasks {
            self.allocate(task.memory_requirement)?;
        }

        Ok(())
    }

    fn deallocate_completed_tasks(&mut self, completed_tasks: &[crate::task_scheduling::Task]) -> Result<(), MemoryError> {
        for task in completed_tasks {
            self.deallocate(task.memory_requirement)?;
        }
        Ok(())
    }

    fn force_free(&mut self, size: usize) -> Result<(), MemoryError> {
        let blocks_needed = (size + self.block_size - 1) / self.block_size;
        let total_size = blocks_needed * self.block_size;

        if total_size > self.allocated_memory {
            return Err(MemoryError::InsufficientMemory);
        }

        self.allocated_memory -= total_size;

        // Remove the freed memory from the pool
        for (_, addresses) in self.memory_pool.iter_mut() {
            addresses.retain(|&addr| addr < self.allocated_memory);
        }

        // Remove any empty entries from the memory pool
        self.memory_pool.retain(|_, addresses| !addresses.is_empty());

        Ok(())
    }
}

// Remove duplicate implementation

pub enum MemoryStrategy {
    Simple(SimpleMemoryManager),
    Dynamic(DynamicMemoryManager),
}

impl MemoryStrategy {
    pub fn get_available_memory(&self) -> usize {
        match self {
            MemoryStrategy::Simple(manager) => manager.get_available_memory(),
            MemoryStrategy::Dynamic(manager) => manager.get_available_memory(),
        }
    }
}

impl MemoryManager for MemoryStrategy {
    fn allocate(&mut self, size: usize) -> Result<(), MemoryError> {
        match self {
            MemoryStrategy::Simple(manager) => manager.allocate(size),
            MemoryStrategy::Dynamic(manager) => manager.allocate(size),
        }
    }

    fn deallocate(&mut self, size: usize) -> Result<(), MemoryError> {
        match self {
            MemoryStrategy::Simple(manager) => manager.deallocate(size),
            MemoryStrategy::Dynamic(manager) => manager.deallocate(size),
        }
    }

    fn get_available_memory(&self) -> usize {
        match self {
            MemoryStrategy::Simple(manager) => manager.get_available_memory(),
            MemoryStrategy::Dynamic(manager) => manager.get_available_memory(),
        }
    }

    fn allocate_for_tasks(&mut self, tasks: &[crate::task_scheduling::Task]) -> Result<(), MemoryError> {
        match self {
            MemoryStrategy::Simple(manager) => manager.allocate_for_tasks(tasks),
            MemoryStrategy::Dynamic(manager) => manager.allocate_for_tasks(tasks),
        }
    }

    fn deallocate_completed_tasks(&mut self, completed_tasks: &[crate::task_scheduling::Task]) -> Result<(), MemoryError> {
        match self {
            MemoryStrategy::Simple(manager) => manager.deallocate_completed_tasks(completed_tasks),
            MemoryStrategy::Dynamic(manager) => manager.deallocate_completed_tasks(completed_tasks),
        }
    }

    fn force_free(&mut self, size: usize) -> Result<(), MemoryError> {
        match self {
            MemoryStrategy::Simple(manager) => manager.force_free(size),
            MemoryStrategy::Dynamic(manager) => manager.force_free(size),
        }
    }
}

// The force_free implementations for SimpleMemoryManager and DynamicMemoryManager
// have been removed from this location as they are already correctly implemented
// earlier in the file. This avoids duplication and potential confusion.
