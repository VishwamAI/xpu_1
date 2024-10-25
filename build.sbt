// XPU Hardware Design Build Configuration
name := "xpu-hardware"
version := "0.1.0"
scalaVersion := "2.13.10"

// Chisel Dependencies
libraryDependencies ++= Seq(
  "edu.berkeley.cs" %% "chisel3" % "3.6.0",
  "edu.berkeley.cs" %% "chiseltest" % "0.6.0" % "test"
)

scalacOptions ++= Seq(
  "-language:reflectiveCalls",
  "-deprecation",
  "-feature",
  "-Xcheckinit",
  "-P:chiselplugin:genBundleElements"
)

addCompilerPlugin("edu.berkeley.cs" % "chisel3-plugin" % "3.6.0" cross CrossVersion.full)
