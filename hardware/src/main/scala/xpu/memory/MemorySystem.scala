package xpu.memory

import chisel3._
import chisel3.util._

// Memory System Configuration
class MemoryConfig {
  val numHBMChannels = 8
  val channelWidth = 256  // bits
  val cacheLineSize = 64  // bytes
  val prefetchDepth = 16
}

// HBM Channel Interface
class HBMChannel extends Bundle {
  val addr = Input(UInt(40.W))
  val writeData = Input(UInt(256.W))
  val readData = Output(UInt(256.W))
  val writeEn = Input(Bool())
  val readEn = Input(Bool())
  val ready = Output(Bool())
  val valid = Output(Bool())
}

// Memory Subsystem Module
class MemorySubsystem extends Module {
  val config = new MemoryConfig()

  val io = IO(new Bundle {
    val hbmChannels = Vec(config.numHBMChannels, new HBMChannel())
    val cacheRequest = Flipped(DecoupledIO(new CacheRequest()))
    val cacheResponse = DecoupledIO(new CacheResponse())
    val prefetchAddr = Input(UInt(40.W))
  })

  // Instantiate components
  val hbmController = Module(new HBMController(config))
  val cacheHierarchy = Module(new CacheHierarchy(config))
  val prefetchEngine = Module(new PrefetchEngine(config))

  // Connect HBM channels
  hbmController.io.channels <> io.hbmChannels

  // Connect cache hierarchy
  cacheHierarchy.io.request <> io.cacheRequest
  cacheHierarchy.io.response <> io.cacheResponse
  cacheHierarchy.io.hbmInterface <> hbmController.io.cacheInterface

  // Connect prefetch engine
  prefetchEngine.io.triggerAddr := io.prefetchAddr
  prefetchEngine.io.cacheInterface <> cacheHierarchy.io.prefetchInterface
}

// Basic interface definitions
class CacheRequest extends Bundle {
  val addr = UInt(40.W)
  val data = UInt(512.W)
  val isWrite = Bool()
}

class CacheResponse extends Bundle {
  val data = UInt(512.W)
  val valid = Bool()
}

// Component stubs - to be implemented in separate files
class HBMController(config: MemoryConfig) extends Module {
  val io = IO(new Bundle {
    val channels = Vec(config.numHBMChannels, new HBMChannel())
    val cacheInterface = Flipped(new CacheRequest())
  })
  // Implementation in HBMController.scala
}

class CacheHierarchy(config: MemoryConfig) extends Module {
  val io = IO(new Bundle {
    val request = Flipped(DecoupledIO(new CacheRequest()))
    val response = DecoupledIO(new CacheResponse())
    val hbmInterface = DecoupledIO(new CacheRequest())
    val prefetchInterface = Flipped(DecoupledIO(new CacheRequest()))
  })
  // Implementation in CacheHierarchy.scala
}

class PrefetchEngine(config: MemoryConfig) extends Module {
  val io = IO(new Bundle {
    val triggerAddr = Input(UInt(40.W))
    val cacheInterface = DecoupledIO(new CacheRequest())
  })
  // Implementation in PrefetchEngine.scala
}
