# geometry-aggregation

`geometry-aggregation` is now organized as a research repo about a broader idea:

**set encoders as discretizations of continuum functionals**

The current codebase argues that many set-valued inputs are finite samples from an underlying continuum object, and that pooling over those samples should be analyzed as numerical estimation rather than only as permutation-invariant aggregation.

The concrete claim supported in this repo is:

- uniform averaging behaves like a lower-order estimator and produces roughly `O(1/M)` representation drift under refinement
- geometry-aware weighting behaves like a higher-order estimator and produces roughly `O(1/M^2)` drift in the Darcy 1D case study

Darcy 1D and the trimmed SetONet-style model remain in the repo, but they are now presented as a case study rather than the main identity of the project.

## Repository Layout

```text
geometry-aggregation/
├── set_encoders/
│   ├── encoders.py
│   ├── models.py
│   ├── utils.py
│   └── weights.py
├── case_studies/
│   ├── darcy_1d/
│   │   ├── common.py
│   │   ├── train.py
│   │   ├── sensor_ablation.py
│   │   ├── plot_convergence.py
│   │   └── plot_weight_analysis.py
│   ├── point_cloud_consistency/
│   │   ├── dataset.py
│   │   ├── models.py
│   │   ├── benchmark.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── plot_consistency.py
│   │   └── run_benchmark.py
│   └── sphere_signal_reconstruction/
│       ├── dataset.py
│       ├── models.py
│       ├── benchmark.py
│       ├── train.py
│       ├── evaluate.py
│       ├── plot_summary.py
│       ├── plot_convergence.py
│       ├── plot_qualitative.py
│       └── run_benchmark.py
├── data/
│   └── darcy_1d/
├── checkpoints/
│   ├── setonet_key_trapezoidal/
│   └── setonet_key_uniform/
├── experiments/
│   └── ... compatibility wrappers for old entry points
├── geometry_aggregation/
│   └── ... compatibility wrappers for old imports
├── ABSTRACT.md
└── results/
```

## Core Files

- `set_encoders/weights.py`
  - geometry-aware aggregation weights inferred from sampled coordinates
- `set_encoders/encoders.py`
  - weighted additive set encoder for sampled continuum data
- `set_encoders/models.py`
  - minimal operator-learning model built around the set encoder
- `case_studies/darcy_1d/plot_convergence.py`
  - synthetic convergence plot and actual encoder-drift plot
- `case_studies/darcy_1d/sensor_ablation.py`
  - Darcy 1D sensor-count ablation
- `case_studies/darcy_1d/plot_weight_analysis.py`
  - branch drift, pointwise error, and per-sample error analysis
- `case_studies/point_cloud_consistency/dataset.py`
  - same-object synthetic point-cloud benchmark with controlled density shifts
- `case_studies/point_cloud_consistency/models.py`
  - matched uniform and geometry-aware set classifiers
- `case_studies/point_cloud_consistency/run_benchmark.py`
  - trains both models, evaluates resampling robustness, and saves metrics
- `case_studies/sphere_signal_reconstruction/dataset.py`
  - analytic sphere-field reconstruction benchmark with dense canonical queries
- `case_studies/sphere_signal_reconstruction/run_benchmark.py`
  - trains both reconstruction models, evaluates density shift and deterministic refinement, and saves figures

## Install

```bash
pip install -r requirements.txt
```

## Reproduce The Darcy 1D Case Study

From the repository root:

```bash
make reproduce
```

This generates:

- `results/darcy_1d_sensor_ablation.png`
- `results/quadrature_convergence.png`
- `results/weight_analysis.png`

## Run Individual Case-Study Scripts

Sensor-count ablation:

```bash
python case_studies/darcy_1d/sensor_ablation.py --device cuda:0
```

Continuum-functional convergence and encoder-drift plot:

```bash
python case_studies/darcy_1d/plot_convergence.py --device cuda:0 --n_test 200
```

Detailed weight analysis:

```bash
python case_studies/darcy_1d/plot_weight_analysis.py --device cuda:0 --n_test 200 --sample_idx 5
```

Train the geometry-aware case-study model:

```bash
python case_studies/darcy_1d/train.py \
  --device cuda:0 \
  --output_dir checkpoints/run_geometry_aware
```

Train the uniform-weight control:

```bash
python case_studies/darcy_1d/train.py \
  --device cuda:0 \
  --uniform_weights \
  --output_dir checkpoints/run_uniform
```

## Run The Synthetic Point-Cloud Consistency Benchmark

This benchmark is the first repo-native experiment aimed directly at the paper thesis rather than a single operator-learning architecture. It constructs continuous scalar fields on a sphere, resamples the same underlying object under different point counts and density shifts, and compares:

- a matched uniform set encoder
- a geometry-aware encoder with kNN density weights

Run the full benchmark:

```bash
python case_studies/point_cloud_consistency/run_benchmark.py \
  --device cuda:0 \
  --output_dir results/point_cloud_consistency_run
```

Train a single model:

```bash
python case_studies/point_cloud_consistency/train.py \
  --device cuda:0 \
  --weight_mode knn \
  --output_dir checkpoints/point_cloud_consistency/geometry_aware
```

Evaluate two checkpoints:

```bash
python case_studies/point_cloud_consistency/evaluate.py \
  --device cuda:0 \
  --uniform_checkpoint checkpoints/point_cloud_consistency/uniform \
  --geometry_checkpoint checkpoints/point_cloud_consistency/geometry_aware \
  --output_path results/point_cloud_consistency_metrics.json
```

Plot saved metrics:

```bash
python case_studies/point_cloud_consistency/plot_consistency.py \
  --metrics_path results/point_cloud_consistency_metrics.json \
  --output_dir results
```

## Run The Point-Cloud Mean-Regression Benchmark

This task uses the same synthetic sphere dataset, but predicts the object's
continuum-average scalar value directly instead of thresholding it into a
binary class. That makes the benchmark align more directly with the numerical
estimation story in the paper.

Run the full regression benchmark:

```bash
python case_studies/point_cloud_consistency/run_mean_regression.py \
  --device cuda:0 \
  --output_dir results/point_cloud_mean_regression_run
```

Train a single regression model:

```bash
python case_studies/point_cloud_consistency/train_regression.py \
  --device cuda:0 \
  --weight_mode knn \
  --output_dir checkpoints/point_cloud_mean_regression/geometry_aware
```

Evaluate two regression checkpoints:

```bash
python case_studies/point_cloud_consistency/evaluate_regression.py \
  --device cuda:0 \
  --uniform_checkpoint checkpoints/point_cloud_mean_regression/uniform \
  --geometry_checkpoint checkpoints/point_cloud_mean_regression/geometry_aware \
  --output_path results/point_cloud_mean_regression_metrics.json
```

Plot saved regression metrics:

```bash
python case_studies/point_cloud_consistency/plot_regression.py \
  --metrics_path results/point_cloud_mean_regression_metrics.json \
  --output_dir results
```

## Run The Sphere Signal Reconstruction Benchmark

This benchmark predicts a dense scalar field on a fixed canonical sphere query
set from sparse observed sphere samples. It is designed to make
discretization-shift robustness visible both qualitatively and through
deterministic refinement curves.

Run the full benchmark:

```bash
python case_studies/sphere_signal_reconstruction/run_benchmark.py \
  --device cuda:0 \
  --output_dir results/sphere_signal_reconstruction_run
```

Train a single reconstruction model:

```bash
python case_studies/sphere_signal_reconstruction/train.py \
  --device cuda:0 \
  --weight_mode knn \
  --output_dir checkpoints/sphere_signal_reconstruction/geometry_aware
```

Evaluate two checkpoints:

```bash
python case_studies/sphere_signal_reconstruction/evaluate.py \
  --device cuda:0 \
  --uniform_checkpoint checkpoints/sphere_signal_reconstruction/uniform \
  --geometry_checkpoint checkpoints/sphere_signal_reconstruction/geometry_aware \
  --output_path results/sphere_signal_reconstruction_metrics.json
```

The full run writes:

- `results/sphere_signal_reconstruction_run/metrics.json`
- `results/sphere_signal_reconstruction_run/sphere_signal_reconstruction_summary.png`
- `results/sphere_signal_reconstruction_run/sphere_signal_reconstruction_convergence.png`
- `results/sphere_signal_reconstruction_run/sphere_signal_reconstruction_qualitative.png`

## Pretrained Checkpoints

The repository ships with two Darcy 1D checkpoints:

- `checkpoints/setonet_key_trapezoidal/`
- `checkpoints/setonet_key_uniform/`

These remain named after the original SetONet-style case study for compatibility with existing artifacts.

## Dataset

The bundled Darcy 1D HuggingFace dataset lives under:

- `data/darcy_1d/darcy_1d_dataset_501/`

The loader is:

- `data/darcy_1d/dataset.py`

## Compatibility

The old modules and scripts are still present as thin wrappers:

- `geometry_aggregation/` re-exports the new implementation for older imports
- `experiments/` points to the new Darcy case-study entry points

That keeps the current repo usable while shifting the conceptual center of gravity toward general set encoding rather than a single architecture name.

## Project Direction

The intended direction is broader than operator learning. The repo is being shaped around a foundational thesis:

- set encoders over sampled geometric or measure-supported data should be judged by discretization consistency
- permutation invariance is necessary but not sufficient
- weighting rules are numerical estimators, not just implementation details

That framing is captured in `ABSTRACT.md`. The repo now contains:

- a Darcy 1D operator-learning case study
- a synthetic same-object point-cloud consistency benchmark for controlled resampling shift
- a dense sphere-field reconstruction benchmark for discretization shift and refinement
