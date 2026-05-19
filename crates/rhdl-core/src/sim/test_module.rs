//! Lightweight test module for using Icarus Verilog
use crate::RHDLError;
use rhdl_vlog as vlog;

/// A simple test module that can be used to run Verilog simulations
/// using Icarus Verilog.
pub struct TestModule(vlog::ModuleList);

impl From<vlog::ModuleList> for TestModule {
    fn from(m: vlog::ModuleList) -> Self {
        Self(m)
    }
}

impl std::fmt::Display for TestModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl TestModule {
    /// Run the test module using explicit paths to the Icarus Verilog
    /// `iverilog` and `vvp` binaries. `ivl_base` is the IVL_BASE
    /// directory iverilog uses to find its `*.tgt` modules; pass an
    /// empty path to rely on iverilog's own install-time discovery.
    ///
    /// The test module should include a test bench that prints
    /// "TESTBENCH OK" on success, and "FAILED" on failure.
    pub fn run_iverilog_with(
        &self,
        iverilog: &std::path::Path,
        vvp: &std::path::Path,
        ivl_base: &std::path::Path,
    ) -> Result<(), RHDLError> {
        let d = tempfile::tempdir()?;
        let d_path = d.path();
        std::fs::write(d_path.join("testbench.v"), self.to_string())?;
        let mut cmd = std::process::Command::new(iverilog);
        if !ivl_base.as_os_str().is_empty() {
            cmd.env("IVL_BASE", ivl_base);
        }
        cmd.arg("-o")
            .arg(d_path.join("testbench"))
            .arg(d_path.join("testbench.v"));
        let status = cmd.status().map_err(|e| {
            anyhow::anyhow!("Failed to invoke iverilog at {}: {e}", iverilog.display())
        })?;
        if !status.success() {
            return Err(anyhow::anyhow!("Failed to compile testbench with {}", status).into());
        }
        let mut cmd = std::process::Command::new(vvp);
        cmd.arg(d_path.join("testbench"));
        let output = cmd
            .output()
            .map_err(|e| anyhow::anyhow!("Failed to invoke vvp at {}: {e}", vvp.display()))?;
        let output_stdout = String::from_utf8_lossy(&output.stdout);
        for line in output_stdout.lines() {
            if line.contains("FAILED") {
                return Err(RHDLError::VerilogVerificationErrorString(line.into()));
            }
            if line.starts_with("TESTBENCH OK") {
                return Ok(());
            }
        }
        Err(RHDLError::VerilogVerificationErrorString(
            "No output".into(),
        ))
    }

    /// Back-compat wrapper assuming `iverilog` and `vvp` are on `$PATH`.
    /// Prefer [`Self::run_iverilog_with`] in any sandboxed / hermetic
    /// context.
    pub fn run_iverilog(&self) -> Result<(), RHDLError> {
        self.run_iverilog_with(
            std::path::Path::new("iverilog"),
            std::path::Path::new("vvp"),
            std::path::Path::new(""),
        )
    }
}
