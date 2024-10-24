// Number Theoretic Transform Implementation for Post-Quantum Cryptography
package xpu.security

import chisel3._
import chisel3.util._

// Kyber Parameters
class KyberConfig {
  val k = 3  // Security parameter (Kyber768)
  val n = 256  // Polynomial degree
  val q = 3329  // Modulus
  val eta1 = 2  // Noise parameter
  val eta2 = 2  // Noise parameter
}

// Number Theoretic Transform (NTT) for polynomial operations
class NTTProcessor extends Module {
  val config = new KyberConfig()

  val io = IO(new Bundle {
    val dataIn = Input(Vec(config.n, UInt(13.W)))  // q < 2^13
    val dataOut = Output(Vec(config.n, UInt(13.W)))
    val inverse = Input(Bool())  // true for inverse NTT
    val start = Input(Bool())
    val done = Output(Bool())
  })

  val state = RegInit(0.U(2.W))
  val counter = RegInit(0.U(9.W))
  val data = Reg(Vec(config.n, UInt(13.W)))

  // NTT constants (pre-computed roots of unity)
  val nttConstants = VecInit(Seq.fill(config.n)(0.U(13.W)))

  // Butterfly operation for NTT
  def butterfly(a: UInt, b: UInt, w: UInt): (UInt, UInt) = {
    val temp = (b * w) % config.q.U
    val aNew = (a + temp) % config.q.U
    val bNew = (a - temp + config.q.U) % config.q.U
    (aNew, bNew)
  }

  // NTT state machine
  when(io.start) {
    state := 1.U
    data := io.dataIn
    counter := 0.U
  }.elsewhen(state === 1.U) {
    when(counter < config.n.U) {
      val (a, b) = butterfly(data(counter), data(counter + 1.U), nttConstants(counter))
      data(counter) := a
      data(counter + 1.U) := b
      counter := counter + 2.U
    }.otherwise {
      state := 2.U
    }
  }.elsewhen(state === 2.U) {
    io.done := true.B
    state := 0.U
  }

  io.dataOut := data
}
