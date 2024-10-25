// Secure Network Protocol Implementation for Mesh Network
package xpu.security

import chisel3._
import chisel3.util._

// Secure Network Protocol Module
class SecureNetworkProtocol extends Module {
  val io = IO(new Bundle {
    // Node Interface
    val nodeId = Input(UInt(8.W))
    val numNodes = Input(UInt(8.W))

    // Network Interface
    val sendData = Input(Bool())
    val receiveData = Input(Bool())
    val targetNode = Input(UInt(8.W))
    val dataIn = Input(Vec(512, UInt(8.W)))
    val dataOut = Output(Vec(512, UInt(8.W)))

    // Status Interface
    val busy = Output(Bool())
    val done = Output(Bool())
    val error = Output(Bool())

    // Session Keys
    val sessionKeyValid = Output(Bool())
    val sessionKeyExpired = Output(Bool())
  })

  // Instantiate Kyber KEM
  val kyberKEM = Module(new KyberKEM())

  // Session key storage
  val sessionKeys = Mem(256, Vec(32, UInt(8.W)))  // Store 256-bit keys for up to 256 nodes
  val keyTimestamps = Reg(Vec(256, UInt(32.W)))   // Store key creation timestamps
  val keyTimeout = 3600.U  // Key validity period (1 hour in seconds)

  // Protocol States
  val sIdle :: sKeyEstablishment :: sEncrypt :: sDecrypt :: Nil = Enum(4)
  val state = RegInit(sIdle)

  // Session management
  val currentTime = RegInit(0.U(32.W))
  currentTime := currentTime + 1.U  // Simple timer, would be real timestamp in production

  // Check if session key needs renewal
  val needKeyRenewal = WireDefault(false.B)
  when(io.targetNode < io.numNodes) {
    needKeyRenewal := (currentTime - keyTimestamps(io.targetNode)) >= keyTimeout
  }

  // Key establishment process
  when(needKeyRenewal || !io.sessionKeyValid) {
    // Initiate key exchange
    kyberKEM.io.generateKey := true.B
    when(kyberKEM.io.done) {
      // Store session key and timestamp
      sessionKeys(io.targetNode) := kyberKEM.io.sharedSecret
      keyTimestamps(io.targetNode) := currentTime
    }
  }

  // Main state machine
  switch(state) {
    is(sIdle) {
      when(io.sendData) {
        when(needKeyRenewal) {
          state := sKeyEstablishment
        }.otherwise {
          state := sEncrypt
        }
      }.elsewhen(io.receiveData) {
        state := sDecrypt
      }
    }

    is(sKeyEstablishment) {
      when(kyberKEM.io.done) {
        state := sEncrypt
      }
    }

    is(sEncrypt) {
      // Encrypt data using session key
      // In practice, would use AES-GCM or similar
      val sessionKey = sessionKeys(io.targetNode)
      for (i <- 0 until 512) {
        io.dataOut(i) := io.dataIn(i) ^ sessionKey(i % 32)
      }
      state := sIdle
    }

    is(sDecrypt) {
      // Decrypt data using session key
      val sessionKey = sessionKeys(io.nodeId)
      for (i <- 0 until 512) {
        io.dataOut(i) := io.dataIn(i) ^ sessionKey(i % 32)
      }
      state := sIdle
    }
  }

  // Status outputs
  io.busy := state =/= sIdle
  io.done := state === sIdle && RegNext(state =/= sIdle)
  io.error := false.B
  io.sessionKeyValid := keyTimestamps(io.targetNode) > 0.U && !needKeyRenewal
  io.sessionKeyExpired := needKeyRenewal

  // Connect Kyber KEM
  kyberKEM.io.encapsulate := false.B
  kyberKEM.io.decapsulate := false.B
  when(state === sIdle) {
    kyberKEM.io.generateKey := false.B
  }
}
