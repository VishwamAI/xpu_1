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
}

#[derive(Debug, Clone, Copy)]
pub enum MemoryManagerType {
    Simple,
    Dynamic,
}

pub struct SimpleMemoryManager {
    memory_pool: BTreeMap<usize, Vec<usize>>,  // Maps block size to list of block addresses
    allocated_blocks: BTreeMap<usize, usize>,   // Maps block address to block size
    total_memory: usize,
    allocated_memory: usize,
    fragmentation_threshold: f32,
}

impl SimpleMemoryManager {
    pub fn new(total_memory: usize) -> Self {
        SimpleMemoryManager {
            memory_pool: BTreeMap::new(),
            allocated_blocks: BTreeMap::new(),
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

        let mut _allocated = false;
        let mut block_to_remove = None;
        let mut block_addr_to_use = None;
        let mut original_block_size = 0;

        // First try to find an existing free block of suitable size
        for (block_size, blocks) in self.memory_pool.range_mut(size..) {
            if let Some(block_addr) = blocks.last() {
                block_addr_to_use = Some(*block_addr);
                original_block_size = *block_size;
                if blocks.len() == 1 {
                    block_to_remove = Some(*block_size);
                }
                _allocated = true;
                break;
            }
        }

        if let Some(block_addr) = block_addr_to_use {
            // Remove the block from the pool
            if let Some(size_to_remove) = block_to_remove {
                self.memory_pool.remove(&size_to_remove);
            } else if let Some(blocks) = self.memory_pool.get_mut(&original_block_size) {
                blocks.pop();
            }

            // Record the allocation
            self.allocated_blocks.insert(block_addr, size);
            self.allocated_memory += size;

            // If the block is larger than needed, create a new free block
            if original_block_size > size {
                let remaining = original_block_size - size;
                let remaining_addr = block_addr + size;
                self.memory_pool
                    .entry(remaining)
                    .or_default()
                    .push(remaining_addr);
            }
        } else {
            // If no suitable block found, allocate from free memory
            let block_addr = self.allocated_memory;
            self.allocated_blocks.insert(block_addr, size);
            self.allocated_memory += size;
        }

        if self.fragmentation_level() > self.fragmentation_threshold {
            self.defragment();
        }

        Ok(())
    }

    fn deallocate(&mut self, size: usize) -> Result<(), MemoryError> {
        // Find the block address for this size
        if let Some((&block_addr, &_block_size)) = self.allocated_blocks
            .iter()
            .find(|(_, &s)| s == size)
        {
            // Remove from allocated blocks
            self.allocated_blocks.remove(&block_addr);

            // Add back to memory pool
            self.memory_pool
                .entry(size)
                .or_default()
                .push(block_addr);

            // Update allocated memory
            self.allocated_memory = self.allocated_memory.saturating_sub(size);

            // Attempt to merge adjacent free blocks if possible
            if self.fragmentation_level() > self.fragmentation_threshold {
                self.defragment();
            }

            Ok(())
        } else {
            Err(MemoryError::BlockNotFound)
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
            if let Some(_freed_address) = addresses.pop() {
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
}

pub enum MemoryStrategy {
    Simple(SimpleMemoryManager),
    Dynamic(DynamicMemoryManager),
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
}
