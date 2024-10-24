// Dynamic Voltage and Frequency Scaling Controller
package xpu.power

import chisel3._
import chisel3.util._

class DVFSController(config: PowerConfig) extends Module {
  val io = IO(new Bundle {
    val workloadStats = Input(new WorkloadStats())
    val powerState = Output(new PowerState())
    val targetPerformance = Input(UInt(8.W))  // 0-100%
    val workloadType = Input(UInt(2.W))       // 0: LLM, 1: Robotics, 2: Mixed
  })

  // DVFS state machine states
  val sIdle :: sMonitoring :: sAdjusting :: Nil = Enum(3)
  val state = RegInit(sIdle)

  // Monitoring counters and accumulators
  val monitorCounter = RegInit(0.U(32.W))
  val utilizationAccum = RegInit(0.U(32.W))
  val ipcAccum = RegInit(0.U(32.W))
  val memBandwidthAccum = RegInit(0.U(32.W))

  // Current power state
  val currentState = RegInit(0.U.asTypeOf(new PowerState))

  // Workload-specific thresholds
  val utilizationThreshold = WireDefault(10.U)
  val monitoringPeriod = WireDefault(config.monitoringPeriod.U)

  // Set workload-specific parameters
  switch(io.workloadType) {
    is(0.U) { // LLM workload - more aggressive power saving
      utilizationThreshold := 15.U
      monitoringPeriod := config.monitoringPeriod.U * 2.U
    }
    is(1.U) { // Robotics workload - faster response
      utilizationThreshold := 5.U
      monitoringPeriod := config.monitoringPeriod.U / 2.U
    }
    is(2.U) { // Mixed workload - balanced approach
      utilizationThreshold := 10.U
      monitoringPeriod := config.monitoringPeriod.U
    }
  }

  // DVFS control logic
  switch(state) {
    is(sIdle) {
      when(io.workloadStats.utilization =/= currentState.frequency) {
        state := sMonitoring
        monitorCounter := 0.U
        utilizationAccum := 0.U
        ipcAccum := 0.U
        memBandwidthAccum := 0.U
      }
    }

    is(sMonitoring) {
      monitorCounter := monitorCounter + 1.U
      utilizationAccum := utilizationAccum + io.workloadStats.utilization
      ipcAccum := ipcAccum + io.workloadStats.ipcRate
      memBandwidthAccum := memBandwidthAccum + io.workloadStats.memoryBandwidth

      when(monitorCounter === monitoringPeriod) {
        state := sAdjusting
      }
    }

    is(sAdjusting) {
      val avgUtilization = utilizationAccum / monitoringPeriod
      val avgIpc = ipcAccum / monitoringPeriod
      val avgMemBandwidth = memBandwidthAccum / monitoringPeriod

      // Workload-specific DVFS decisions
      when(io.workloadType === 0.U) { // LLM workload
        // Prioritize power efficiency over performance
        when(avgUtilization > currentState.frequency + utilizationThreshold &&
             avgMemBandwidth > (config.memoryThreshold.U / 2.U)) {
          when(currentState.frequency < (config.numFrequencySteps - 1).U) {
            currentState.frequency := currentState.frequency + 1.U
            currentState.voltage := currentState.voltage + 1.U
          }
        }.elsewhen(avgUtilization < currentState.frequency - utilizationThreshold) {
          when(currentState.frequency > 0.U) {
            currentState.frequency := currentState.frequency - 1.U
            currentState.voltage := currentState.voltage - 1.U
          }
        }
      }.elsewhen(io.workloadType === 1.U) { // Robotics workload
        // Prioritize performance and responsiveness
        when(avgUtilization > currentState.frequency + utilizationThreshold ||
             avgIpc > config.ipcThreshold.U) {
          when(currentState.frequency < (config.numFrequencySteps - 1).U) {
            currentState.frequency := currentState.frequency + 2.U  // Faster ramp-up
            currentState.voltage := currentState.voltage + 2.U
          }
        }.elsewhen(avgUtilization < currentState.frequency - utilizationThreshold * 2.U) {
          when(currentState.frequency > 0.U) {
            currentState.frequency := currentState.frequency - 1.U
            currentState.voltage := currentState.voltage - 1.U
          }
        }
      }.otherwise { // Mixed workload
        // Balanced approach
        when(avgUtilization > currentState.frequency + utilizationThreshold) {
          when(currentState.frequency < (config.numFrequencySteps - 1).U) {
            currentState.frequency := currentState.frequency + 1.U
            currentState.voltage := currentState.voltage + 1.U
          }
        }.elsewhen(avgUtilization < currentState.frequency - utilizationThreshold) {
          when(currentState.frequency > 0.U) {
            currentState.frequency := currentState.frequency - 1.U
            currentState.voltage := currentState.voltage - 1.U
          }
        }
      }

      // Calculate current power consumption based on V/F setting
      currentState.currentPower := (currentState.voltage * currentState.voltage *
                                  currentState.frequency * config.powerScaleFactor.U)

      state := sIdle
    }
  }

  // Connect output
  io.powerState := currentState
}
