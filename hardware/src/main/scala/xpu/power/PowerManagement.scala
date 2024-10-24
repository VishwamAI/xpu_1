// Power Management Unit implementation for XPU Architecture
package xpu.power

import chisel3._
import chisel3.util._

// Workload Statistics Bundle
class WorkloadStats extends Bundle {
  val utilization = UInt(8.W)        // 0-100%
  val temperature = UInt(8.W)        // Celsius
  val ipcRate = UInt(8.W)           // Instructions per cycle
  val memoryBandwidth = UInt(16.W)  // MB/s
}

// Power Management Configuration
class PowerConfig {
  val numVoltageSteps = 8
  val numFrequencySteps = 16
  val numPowerDomains = 4  // Transformer, Memory, Robotics, Network
  val monitoringPeriod = 1000  // cycles
  val idleThreshold = 100      // cycles
  val memoryThreshold = 1024   // MB/s
  val ipcThreshold = 50        // IPC threshold for robotics
  val powerScaleFactor = 2     // mW scaling factor
  val tempThreshold = 85       // Celsius
}

// Power Domain State
class PowerState extends Bundle {
  val voltage = UInt(3.W)      // 8 voltage levels
  val frequency = UInt(4.W)    // 16 frequency levels
  val powerGated = Bool()      // Power gating status
  val currentPower = UInt(16.W)// Power consumption in mW
  val temperature = UInt(8.W)  // Current temperature
}

// Main Power Management Unit
class PowerManagementUnit extends Module {
  val config = new PowerConfig()

  val io = IO(new Bundle {
    val domainStats = Vec(config.numPowerDomains, Input(new WorkloadStats()))
    val powerStates = Vec(config.numPowerDomains, Output(new PowerState()))
    val totalPower = Output(UInt(32.W))
    val powerSavingMode = Input(Bool())
    val clockEnable = Vec(config.numPowerDomains, Output(Bool()))
    val workloadTypes = Vec(config.numPowerDomains, Input(UInt(2.W))) // 0: LLM, 1: Robotics, 2: Mixed
  })

  // Instantiate DVFS and power gating controllers for each domain
  val dvfsControllers = Seq.fill(config.numPowerDomains)(Module(new DVFSController(config)))
  val powerGatingControllers = Seq.fill(config.numPowerDomains)(Module(new PowerGatingController(config)))

  // Power state registers for each domain
  val domainPowerStates = RegInit(VecInit(Seq.fill(config.numPowerDomains)(
    0.U.asTypeOf(new PowerState)
  )))

  // Temperature-based throttling states
  val thermalThrottling = RegInit(VecInit(Seq.fill(config.numPowerDomains)(false.B)))

  // Connect controllers for each domain
  for (i <- 0 until config.numPowerDomains) {
    // Connect workload stats and type to controllers
    dvfsControllers(i).io.workloadStats := io.domainStats(i)
    dvfsControllers(i).io.workloadType := io.workloadTypes(i)
    powerGatingControllers(i).io.workloadStats := io.domainStats(i)
    powerGatingControllers(i).io.domainType := i.U

    // Set target performance based on workload type and power saving mode
    val basePerformance = WireDefault(100.U)
    when(io.powerSavingMode) {
      basePerformance := 50.U
    }.elsewhen(io.workloadTypes(i) === 0.U) { // LLM workload
      basePerformance := 75.U
    }.elsewhen(io.workloadTypes(i) === 1.U) { // Robotics workload
      basePerformance := 100.U
    }

    dvfsControllers(i).io.targetPerformance := basePerformance

    // Temperature-based throttling
    when(io.domainStats(i).temperature >= config.tempThreshold.U) {
      thermalThrottling(i) := true.B
    }.elsewhen(io.domainStats(i).temperature <= (config.tempThreshold - 10).U) {
      thermalThrottling(i) := false.B
    }

    // Update domain power states with thermal consideration
    when(thermalThrottling(i)) {
      domainPowerStates(i).frequency := (dvfsControllers(i).io.powerState.frequency >> 1)
      domainPowerStates(i).voltage := (dvfsControllers(i).io.powerState.voltage >> 1)
    }.elsewhen(!powerGatingControllers(i).io.powerGateEnable) {
      domainPowerStates(i) := dvfsControllers(i).io.powerState
    }.otherwise {
      domainPowerStates(i).powerGated := true.B
      domainPowerStates(i).currentPower := 0.U
    }

    // Update temperature in power state
    domainPowerStates(i).temperature := io.domainStats(i).temperature

    // Set clock enable based on power gating and thermal throttling
    io.clockEnable(i) := !powerGatingControllers(i).io.powerGateEnable && !thermalThrottling(i)
  }

  // Connect power states to output
  for (i <- 0 until config.numPowerDomains) {
    io.powerStates(i) := domainPowerStates(i)
  }

  // Calculate total power consumption with dynamic and static components
  val dynamicPower = domainPowerStates.map(state =>
    Mux(state.powerGated,
      0.U,
      (state.voltage * state.voltage * state.frequency * config.powerScaleFactor.U)
    )
  ).reduce(_ + _)

  val staticPower = domainPowerStates.map(state =>
    Mux(state.powerGated,
      (state.voltage * config.powerScaleFactor.U >> 3), // Reduced static power in power gated state
      (state.voltage * config.powerScaleFactor.U >> 2)  // Normal static power
    )
  ).reduce(_ + _)

  io.totalPower := dynamicPower + staticPower
}

