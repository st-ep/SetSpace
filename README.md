# Set Encoders as Discretizations of Continuum Functionals

Many set-valued inputs are finite samples from an underlying continuum object, and pooling over those samples should be analyzed as numerical estimation rather than only as permutation-invariant aggregation.

This repo argues that:

- uniform averaging is a low-order estimator whose output drifts under resampling, refinement, or density shift
- geometry-aware weighting corrects for sampling bias and produces more stable representations

## Repository Layout

```text
set-space/
├── set_encoders/
│   ├── encoders.py
│   ├── models.py
│   ├── utils.py
│   └── weights.py
├── case_studies/
│   ├── shared.py
│   ├── sphere_utils.py
│   ├── airfrans_field_prediction/
│   │   ├── prepare_dataset.py
│   │   ├── dataset.py
│   │   ├── models.py
│   │   ├── benchmark.py
│   │   └── run_benchmark.py
│   ├── ahmedml_surface_forces/
│   │   ├── prepare_dataset.py
│   │   ├── dataset.py
│   │   ├── models.py
│   │   ├── benchmark.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── run_benchmark.py
│   ├── point_cloud_consistency/
│   │   ├── dataset.py
│   │   ├── models.py
│   │   ├── benchmark.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── plot_consistency.py
│   │   └── run_benchmark.py
│   ├── sphere_signal_reconstruction/
│   │   ├── dataset.py
│   │   ├── models.py
│   │   ├── benchmark.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── plot_summary.py
│   │   ├── plot_convergence.py
│   │   ├── plot_qualitative.py
│   │   └── run_benchmark.py
├── ABSTRACT.md
└── results/
```

## Core Files

- `set_encoders/weights.py`
  - geometry-aware aggregation weights inferred from sampled coordinates
- `set_encoders/encoders.py`
  - weighted additive set encoder for sampled continuum data
- `set_encoders/models.py`
  - operator-learning model built around the set encoder
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
- `case_studies/ahmedml_surface_forces/prepare_dataset.py`
  - converts raw AhmedML surface files into compact `.npz` samples for sparse surface-force learning
- `case_studies/ahmedml_surface_forces/run_benchmark.py`
  - trains matched uniform and geometry-aware set regressors on sparse surface samples and evaluates resampling robustness
- `case_studies/airfrans_field_prediction/prepare_dataset.py`
  - downloads and preprocesses the official AirfRANS splits into compact airfoil-boundary force-regression `.npz` samples
- `case_studies/airfrans_field_prediction/run_benchmark.py`
  - trains matched uniform and geometry-aware set regressors for sparse AirfRANS `Cd/Cl` prediction

## Install

```bash
pip install -e .
```

For the AhmedML preprocessing script, install the optional dependency:

```bash
pip install -e .[ahmedml]
```

For AirfRANS preprocessing, install the optional dependency:

```bash
pip install -e .[airfrans]
```

## Run The Synthetic Point-Cloud Consistency Benchmark

Constructs continuous scalar fields on a sphere, resamples the same underlying object under different point counts and density shifts, and compares:

- a matched uniform set encoder
- a geometry-aware encoder with kNN density weights

Run the full benchmark. This defaults to `25000` training steps:

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

Predicts the object's continuum-average scalar value directly instead of
thresholding it into a binary class.

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

Predicts a dense scalar field on a fixed canonical sphere query set from sparse
observed sphere samples. Designed to make discretization-shift robustness
visible both qualitatively and through deterministic refinement curves.

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

## Run The AhmedML Sparse-Surface Force Benchmark

This benchmark targets a more directly practical surface-functional task:
predicting aerodynamic force coefficients from irregular sparse samples of CFD
surface fields. The raw AhmedML release provides per-geometry `boundary_*.vtp`
surface files and force/moment CSVs; the repo converts them into compact
surface-sample `.npz` files first.

Prepare the processed dataset:

```bash
python case_studies/ahmedml_surface_forces/prepare_dataset.py \
  --raw_root data/ahmedml_raw \
  --output_dir data/ahmedml_processed \
  --target_names Cd Cl
```

Run the full benchmark:

```bash
python case_studies/ahmedml_surface_forces/run_benchmark.py \
  --device cuda:0 \
  --processed_root data/ahmedml_processed \
  --output_dir results/ahmedml_surface_forces_run
```

Train a single model:

```bash
python case_studies/ahmedml_surface_forces/train.py \
  --device cuda:0 \
  --processed_root data/ahmedml_processed \
  --weight_mode knn \
  --output_dir checkpoints/ahmedml_surface_forces/geometry_aware
```

Or use `make`:

```bash
make ahmedml-prepare AHMEDML_RAW_ROOT=data/ahmedml_raw AHMEDML_ROOT=data/ahmedml_processed
make ahmedml-surface-forces AHMEDML_ROOT=data/ahmedml_processed
make ahmedml-surface-forces-pointnext AHMEDML_ROOT=data/ahmedml_processed
```

Both Ahmed `make` targets default to `25000` training steps. To override:

```bash
make ahmedml-surface-forces-pointnext AHMEDML_ROOT=data/ahmedml_processed AHMEDML_STEPS=5000
```

## Run The AirfRANS Force-Regression Benchmark

This AirfRANS benchmark reuses the official `full/scarce/reynolds/aoa` splits
but converts each simulation into a sparse airfoil-boundary regression task.
The preprocessing step uses the raw simulation geometry and AirfRANS reference
API to compute aerodynamic force coefficients, then stores:

- sparse boundary coordinates on the airfoil
- local boundary features: pressure, wall shear, and normals
- global regression targets `(Cd, Cl)`

Prepare the processed dataset:

```bash
python case_studies/airfrans_field_prediction/prepare_dataset.py \
  --raw_root data/airfrans_raw \
  --output_dir data/airfrans_processed \
  --task full
```

Run the full benchmark. This defaults to `25000` training steps:

```bash
python case_studies/airfrans_field_prediction/run_benchmark.py \
  --device cuda:0 \
  --processed_root data/airfrans_processed \
  --task full \
  --output_dir results/airfrans_field_prediction_run
```

Run the PointNeXt baseline on the same processed split:

```bash
python case_studies/airfrans_field_prediction/run_benchmark.py \
  --device cuda:0 \
  --processed_root data/airfrans_processed \
  --task full \
  --backbone pointnext \
  --output_dir results/airfrans_field_prediction_pointnext_run
```

Or use `make`:

```bash
make airfrans-prepare AIRFRANS_RAW_ROOT=data/airfrans_raw AIRFRANS_ROOT=data/airfrans_processed
make airfrans-force-regression AIRFRANS_ROOT=data/airfrans_processed
make airfrans-force-regression-pointnext AIRFRANS_ROOT=data/airfrans_processed
```

## Project Direction

The repo is shaped around a foundational thesis:

- set encoders over sampled geometric or measure-supported data should be judged by discretization consistency
- permutation invariance is necessary but not sufficient
- weighting rules are numerical estimators, not just implementation details

That framing is captured in `ABSTRACT.md`. The repo contains:

- a synthetic same-object point-cloud consistency benchmark for controlled resampling shift
- a dense sphere-field reconstruction benchmark for discretization shift and refinement
