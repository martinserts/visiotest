//! The model: YOLO11n in ONNX Runtime (`ort`), on the CPU or CoreML execution provider.
//!
//! This is the only file that talks to `ort`. It loads the `.onnx` file, checks that
//! the graph has the shapes the hand-written pre- and postprocessing expect, and runs
//! one input tensor at a time. Tests never create a [`Model`]: the `.onnx` file is
//! gitignored, and a `Session` would need ONNX Runtime at test time.

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use ort::ep::{self, coreml};
use ort::session::{Session, SessionOutputs};
use ort::value::TensorRef;

use super::preprocess::TENSOR_LEN;
use super::{INPUT_SIZE, NUM_ANCHORS, NUM_CLASSES};

/// The model input, NCHW: batch 1, 3 colour planes, 640×640.
const INPUT_SHAPE: [i64; 4] = [1, 3, INPUT_SIZE as i64, INPUT_SIZE as i64];
/// The model output: batch 1, 84 attributes (4 box + 80 class scores), 8400 anchors.
const OUTPUT_SHAPE: [i64; 3] = [1, 4 + NUM_CLASSES as i64, NUM_ANCHORS as i64];
/// Where the model comes from, for error messages.
const EXPORT_HINT: &str = "model-export/export.py";

/// Which ONNX Runtime execution provider runs the graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Ep {
    /// ONNX Runtime's own CPU kernels.
    Cpu,
    /// Apple's CoreML (Neural Engine / GPU / CPU, CoreML decides per op).
    Coreml,
}

impl Ep {
    /// Name for the status bar and the terminal.
    pub fn label(self) -> &'static str {
        match self {
            Ep::Cpu => "CPU",
            Ep::Coreml => "CoreML",
        }
    }

    /// Text of the status-bar chip.
    pub fn chip(self) -> &'static str {
        match self {
            Ep::Cpu => "EP CPU",
            Ep::Coreml => "EP CoreML",
        }
    }
}

/// A loaded, shape-checked YOLO11n session. `run` needs `&mut self`, so one thread
/// (the detector worker) owns it.
pub struct Model {
    session: Session,
}

/// The outputs of one `run`. They borrow the session's memory, so reading them is free.
pub struct ModelOutput<'s> {
    outputs: SessionOutputs<'s>,
}

impl ModelOutput<'_> {
    /// The `[84 × 8400]` output as one flat slice (attribute-major), **without a copy**.
    pub fn raw(&self) -> Result<&[f32]> {
        let (_, data) = self.outputs[0]
            .try_extract_tensor::<f32>()
            .context("reading the model output")?;
        Ok(data)
    }
}

impl Model {
    /// Loads `path` with the chosen execution provider and checks its input and output.
    pub fn load(path: &Path, ep: Ep) -> Result<Model> {
        // One ONNX Runtime environment per process; every session keeps a handle to it.
        let env = ort::init()
            .with_name("visiotest")
            .build()
            .context("starting ONNX Runtime")?;
        let session = Session::builder(&env)
            .context("creating an ONNX Runtime session builder")?
            .with_execution_providers([provider(ep)])
            .map_err(builder_error)?
            .commit_from_file(path)
            .with_context(|| format!("loading {}", path.display()))?;
        check_outlets("input", session.inputs().len())?;
        check_outlets("output", session.outputs().len())?;
        check_shape("input", &tensor_shape(&session.inputs()[0])?, &INPUT_SHAPE)?;
        check_shape(
            "output",
            &tensor_shape(&session.outputs()[0])?,
            &OUTPUT_SHAPE,
        )?;
        Ok(Model { session })
    }

    /// Runs the model on one `1×3×640×640` input tensor.
    pub fn run(&mut self, input: &[f32]) -> Result<ModelOutput<'_>> {
        // A view over our own buffer: ONNX Runtime reads it in place, no copy.
        let shape = INPUT_SHAPE.map(|d| d as usize);
        let tensor = TensorRef::from_array_view((shape, input)).context("wrapping the input")?;
        // Positional input: `load` checked there is exactly one.
        let outputs = self
            .session
            .run(ort::inputs![tensor])
            .context("running the model")?;
        Ok(ModelOutput { outputs })
    }

    /// Runs once on an all-zero input and returns how long it took. CoreML compiles its
    /// graph on the first run, which can take seconds; doing that here keeps it out of
    /// the live latency numbers.
    pub fn warm_up(&mut self) -> Result<Duration> {
        let zeros = vec![0.0; TENSOR_LEN];
        let t0 = Instant::now();
        self.run(&zeros)?.raw()?;
        Ok(t0.elapsed())
    }
}

/// The execution provider to register. Both fail loudly: by default `ort` silently
/// falls back to the CPU when an EP can't register, and `--ep coreml` would then
/// quietly measure the CPU.
fn provider(ep: Ep) -> ep::ExecutionProviderDispatch {
    match ep {
        Ep::Cpu => ep::CPU::default().build().error_on_failure(),
        Ep::Coreml => ep::CoreML::default()
            // ML Program is the newer CoreML format; it supports more ops than
            // NeuralNetwork, so less of the graph falls back to the CPU.
            .with_model_format(coreml::ModelFormat::MLProgram)
            // Let CoreML pick the Neural Engine, GPU or CPU per op.
            .with_compute_units(coreml::ComputeUnits::All)
            // Our input is always 1×3×640×640; static shapes let CoreML take more ops.
            .with_static_input_shapes(true)
            .build()
            .error_on_failure(),
    }
}

/// Builder errors carry the builder back (`ort::Error<SessionBuilder>`), which isn't
/// `Send + Sync`, so `?` can't turn them into `anyhow::Error`. Keep just the message.
fn builder_error<R>(e: ort::Error<R>) -> anyhow::Error {
    anyhow!("configuring the ONNX Runtime session: {e}")
}

/// The dimensions of a tensor input or output (`-1` for a dynamic dimension).
fn tensor_shape(outlet: &ort::value::Outlet) -> Result<Vec<i64>> {
    match outlet.dtype().tensor_shape() {
        Some(shape) => Ok(shape.to_vec()),
        None => bail!(
            "model value `{}` is not a tensor (re-export with {EXPORT_HINT})",
            outlet.name()
        ),
    }
}

fn check_outlets(what: &str, count: usize) -> Result<()> {
    if count != 1 {
        bail!("model has {count} {what}s, expected exactly 1 (re-export with {EXPORT_HINT})");
    }
    Ok(())
}

/// The hand-written pre/postprocessing assume these exact shapes, so a different
/// export (dynamic batch, another image size, an NMS head) must fail at startup.
fn check_shape(what: &str, got: &[i64], want: &[i64]) -> Result<()> {
    if got != want {
        bail!("model {what} shape is {got:?}, expected {want:?} (re-export with {EXPORT_HINT})");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ep_labels() {
        assert_eq!(Ep::Cpu.label(), "CPU");
        assert_eq!(Ep::Coreml.label(), "CoreML");
        assert_eq!(Ep::Coreml.chip(), "EP CoreML");
    }

    #[test]
    fn model_shapes_match_the_export_contract() {
        assert_eq!(INPUT_SHAPE.iter().product::<i64>() as usize, TENSOR_LEN);
        assert_eq!(OUTPUT_SHAPE, [1, 84, 8400]);
    }

    #[test]
    fn model_check_shape_names_the_export_script() {
        assert!(check_shape("input", &[1, 3, 640, 640], &INPUT_SHAPE).is_ok());
        let err = check_shape("input", &[-1, 3, 640, 640], &INPUT_SHAPE).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("[-1, 3, 640, 640]") && msg.contains(EXPORT_HINT),
            "{msg}"
        );
        assert!(check_outlets("output", 2).is_err());
        assert!(check_outlets("output", 1).is_ok());
    }
}
