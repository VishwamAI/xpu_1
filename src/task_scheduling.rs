use std::collections::VecDeque;
use std::time::Duration;

pub struct TaskScheduler {
    pub tasks: VecDeque<Task>,
    processing_units: Vec<ProcessingUnit>,
}

pub struct Task {
    pub id: usize,
    pub priority: u8,
    pub execution_time: Duration,
    pub memory_requirement: usize,
}

pub struct ProcessingUnit {
    pub id: usize,
    pub current_load: Duration,
}

impl TaskScheduler {
    pub fn new(num_processing_units: usize) -> Self {
        TaskScheduler {
            tasks: VecDeque::new(),
            processing_units: (0..num_processing_units)
                .map(|id| ProcessingUnit { id, current_load: Duration::new(0, 0) })
                .collect(),
        }
    }

    pub fn add_task(&mut self, task: Task) {
        let position = self.tasks.iter().position(|t| t.priority < task.priority);
        match position {
            Some(index) => self.tasks.insert(index, task),
            None => self.tasks.push_back(task),
        }
    }

    pub fn get_next_task(&mut self) -> Option<Task> {
        self.tasks.pop_front()
    }

    pub fn schedule(&mut self) -> Vec<Task> {
        println!("Scheduling tasks...");
        let mut completed_tasks = Vec::new();
        while let Some(task) = self.get_next_task() {
            if let Some(unit) = self.find_available_unit() {
                println!("Executing task {} on processing unit {}", task.id, unit.id);
                unit.current_load += task.execution_time;
                completed_tasks.push(task);
            } else {
                println!("No available processing unit for task {}", task.id);
                self.tasks.push_back(task);
            }
        }
        completed_tasks
    }

    fn find_available_unit(&mut self) -> Option<&mut ProcessingUnit> {
        self.processing_units.iter_mut().min_by_key(|unit| unit.current_load)
    }
}
