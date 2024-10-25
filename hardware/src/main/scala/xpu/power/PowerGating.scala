// Power Gating Controller
package xpu.power

import chisel3._
import chisel3.util._

class PowerGatingController(config: PowerConfig) extends Module {
  val io = IO(new Bundle {
    val workloadStats = Input(new WorkloadStats())
    val powerGateEnable = Output(Bool())
    val idleCounter = Output(UInt(32.W))
    val domainType = Input(UInt(2.W))  // 0: Transformer, 1: Memory, 2: Robotics, 3: Network
  })

  // Idle detection counter
  val idleCounter = RegInit(0.U(32.W))

  // Power gating status
  val isPowerGated = RegInit(false.B)

  // Domain-specific idle thresholds
  val idleThresholds = VecInit(Seq(
    config.idleThreshold.U,     // Transformer cores
    (config.idleThreshold/2).U, // Memory subsystem (more aggressive)
    config.idleThreshold.U * 2.U, // Robotics (less aggressive)
    config.idleThreshold.U      // Network
  ))

  // Idle detection logic with domain-specific thresholds
  when(io.workloadStats.utilization === 0.U) {
    idleCounter := idleCounter + 1.U
  }.otherwise {
    idleCounter := 0.U
  }

  // Power gating control with domain-specific policies
  when(idleCounter >= idleThresholds(io.domainType) && !isPowerGated) {
    // Additional checks for safe power gating
    val canPowerGate = WireInit(true.B)

    // Memory domain specific checks
    when(io.domainType === 1.U) {
      canPowerGate := io.workloadStats.memoryBandwidth === 0.U
    }

    // Robotics domain specific checks
    when(io.domainType === 2.U) {
      canPowerGate := io.workloadStats.ipcRate === 0.U
    }

    when(canPowerGate) {
      isPowerGated := true.B
    }
  }.elsewhen(io.workloadStats.utilization > 0.U && isPowerGated) {
    isPowerGated := false.B
  }

  // Power gating wake-up delay counter
  val wakeUpCounter = RegInit(0.U(8.W))
  when(isPowerGated && io.workloadStats.utilization > 0.U) {
    wakeUpCounter := 10.U  // 10 cycle wake-up delay
  }.elsewhen(wakeUpCounter > 0.U) {
    wakeUpCounter := wakeUpCounter - 1.U
  }

  // Connect outputs
  io.powerGateEnable := isPowerGated && (wakeUpCounter === 0.U)
  io.idleCounter := idleCounter
}
