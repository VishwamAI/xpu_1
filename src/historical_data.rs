use std::time::Duration;
use crate::task_scheduling::ProcessingUnitType;
use crate::task_data::HistoricalTaskData;

pub fn get_test_historical_data() -> Vec<HistoricalTaskData> {
    vec![
        // CPU tasks historical data
        HistoricalTaskData {
            task_id: 101,
            execution_time: Duration::from_secs(5),
            memory_usage: 300,
            unit_type: ProcessingUnitType::CPU,
            priority: 3,
        },
        HistoricalTaskData {
            task_id: 102,
            execution_time: Duration::from_secs(4),
            memory_usage: 250,
            unit_type: ProcessingUnitType::CPU,
            priority: 2,
        },
        // GPU tasks historical data
        HistoricalTaskData {
            task_id: 201,
            execution_time: Duration::from_secs(2),
            memory_usage: 200,
            unit_type: ProcessingUnitType::GPU,
            priority: 1,
        },
        HistoricalTaskData {
            task_id: 202,
            execution_time: Duration::from_secs(3),
            memory_usage: 220,
            unit_type: ProcessingUnitType::GPU,
            priority: 2,
        },
        // NPU tasks historical data
        HistoricalTaskData {
            task_id: 301,
            execution_time: Duration::from_secs(4),
            memory_usage: 400,
            unit_type: ProcessingUnitType::NPU,
            priority: 2,
        },
        HistoricalTaskData {
            task_id: 302,
            execution_time: Duration::from_secs(3),
            memory_usage: 350,
            unit_type: ProcessingUnitType::NPU,
            priority: 1,
        },
    ]
}
