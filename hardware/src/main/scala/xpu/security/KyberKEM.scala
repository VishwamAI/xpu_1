// Kyber Key Encapsulation Mechanism Implementation
package xpu.security

import chisel3._
import chisel3.util._

// Kyber KEM Module
class KyberKEM extends Module {
  val config = new KyberConfig()

  val io = IO(new Bundle {
    // Key Generation Interface
    val generateKey = Input(Bool())
    val publicKey = Output(Vec(config.k * config.n, UInt(13.W)))
    val secretKey = Output(Vec(config.k * config.n, UInt(13.W)))

    // Encapsulation Interface
    val encapsulate = Input(Bool())
    val inputPublicKey = Input(Vec(config.k * config.n, UInt(13.W)))
    val ciphertext = Output(Vec(config.k * config.n, UInt(13.W)))
    val sharedSecret = Output(Vec(32, UInt(8.W)))  // 256-bit shared secret

    // Decapsulation Interface
    val decapsulate = Input(Bool())
    val inputCiphertext = Input(Vec(config.k * config.n, UInt(13.W)))
    val inputSecretKey = Input(Vec(config.k * config.n, UInt(13.W)))
    val decapsulatedSecret = Output(Vec(32, UInt(8.W)))

    val busy = Output(Bool())
    val done = Output(Bool())
  })

  // Instantiate NTT processor
  val nttProcessor = Module(new NTTProcessor())

  // State machine states
  val sIdle :: sKeyGen :: sEncap :: sDecap :: Nil = Enum(4)
  val state = RegInit(sIdle)

  // Noise sampler for polynomial coefficients
  def sampleNoise(eta: Int): Vec[UInt] = {
    val coeffs = Wire(Vec(config.n, UInt(13.W)))
    // CBD sampling implementation
    for (i <- 0 until config.n) {
      val a = PopCount(VecInit(Seq.fill(eta)(LFSR()))) // Count 1s in eta bits
      val b = PopCount(VecInit(Seq.fill(eta)(LFSR()))) // Count 1s in eta bits
      coeffs(i) := (a - b + config.q.U) % config.q.U
    }
    coeffs
  }

  // LFSR for random number generation
  def LFSR(): Bool = {
    val lfsr = RegInit("hACE1".U(16.W))
    val feedback = lfsr(15) ^ lfsr(14) ^ lfsr(12) ^ lfsr(3)
    lfsr := Cat(lfsr(14,0), feedback)
    feedback
  }

  // Key Generation Logic
  when(io.generateKey && state === sIdle) {
    state := sKeyGen
    // Generate matrix A (k x k polynomials)
    val matrixA = Wire(Vec(config.k * config.k * config.n, UInt(13.W)))
    // Generate secret vector s
    val secretVector = VecInit(Seq.fill(config.k)(sampleNoise(config.eta1)))
    // Generate error vector e
    val errorVector = VecInit(Seq.fill(config.k)(sampleNoise(config.eta1)))

    // Compute public key t = As + e
    for (i <- 0 until config.k) {
      nttProcessor.io.dataIn := secretVector(i)
      nttProcessor.io.start := true.B
      // Wait for NTT completion and accumulate results
    }

  }.elsewhen(io.encapsulate && state === sIdle) {
    state := sEncap
    // Encapsulation implementation
    // Sample r from distribution
    val r = VecInit(Seq.fill(config.k)(sampleNoise(config.eta1)))
    // Compute u = A^T r + e1
    // Compute v = t^T r + e2 + encode(m)

  }.elsewhen(io.decapsulate && state === sIdle) {
    state := sDecap
    // Decapsulation implementation
    // Compute m' = decode(v - s^T u)
    // Re-encrypt to verify correctness

  }

  // Default outputs
  io.busy := state =/= sIdle
  io.done := false.B

  // Connect NTT processor
  nttProcessor.io.inverse := false.B
  when(state === sIdle) {
    nttProcessor.io.start := false.B
  }
}
