.PHONY: help point-cloud-consistency point-cloud-consistency-512 point-cloud-consistency-pointnext point-cloud-consistency-512-pointnext point-cloud-mean-regression point-cloud-mean-regression-pointnext sphere-reconstruction ahmedml-surface-forces ahmedml-surface-forces-pointnext ahmedml-prepare airfrans-prepare airfrans-force-regression airfrans-force-regression-pointnext airfrans-field-prediction airfrans-field-prediction-pointnext paper-assets paper install clean

DEVICE ?= cuda:0
AHMEDML_ROOT ?= data/ahmedml_processed
AHMEDML_RAW_ROOT ?= data/ahmedml_raw
AHMEDML_STEPS ?= 25000
AIRFRANS_ROOT ?= data/airfrans_processed
AIRFRANS_RAW_ROOT ?= data/airfrans_raw

help:
	@echo "Usage: make <target> [DEVICE=cuda:0]"
	@echo ""
	@echo "Targets:"
	@echo "  install                      Install package in editable mode"
	@echo "  point-cloud-consistency      Synthetic point-cloud consistency benchmark"
	@echo "  point-cloud-consistency-512  Point-cloud benchmark (train at 512 points)"
	@echo "  point-cloud-consistency-pointnext      Synthetic benchmark with PointNeXt backbone"
	@echo "  point-cloud-consistency-512-pointnext  PointNeXt benchmark (train at 512 points)"
	@echo "  point-cloud-mean-regression  Point-cloud mean-regression benchmark"
	@echo "  point-cloud-mean-regression-pointnext  Mean-regression benchmark with PointNeXt backbone"
	@echo "  sphere-reconstruction        Sphere signal reconstruction benchmark"
	@echo "  ahmedml-prepare             Prepare compact AhmedML .npz files from raw boundary data"
	@echo "  ahmedml-surface-forces      AhmedML sparse-surface force benchmark (25000 steps default)"
	@echo "  ahmedml-surface-forces-pointnext  AhmedML PointNeXt surface-force benchmark (25000 steps default)"
	@echo "  airfrans-prepare            Download/preprocess AirfRANS into compact Cd/Cl force-regression .npz files"
	@echo "  airfrans-force-regression   AirfRANS Cd/Cl force-regression benchmark (25000 steps default)"
	@echo "  airfrans-force-regression-pointnext  AirfRANS PointNeXt Cd/Cl force-regression benchmark"
	@echo "  airfrans-field-prediction   Alias for airfrans-force-regression"
	@echo "  airfrans-field-prediction-pointnext  Alias for airfrans-force-regression-pointnext"
	@echo "  paper-assets                 Regenerate all paper figures from current benchmark metrics"
	@echo "  paper                        Regenerate figures, tables, and compile the paper PDF"
	@echo "  clean                        Remove results, logs, and __pycache__"

install:
	pip install -e .

point-cloud-consistency:
	python case_studies/point_cloud_consistency/run_benchmark.py --device $(DEVICE) --output_dir results/point_cloud_consistency_run

point-cloud-consistency-512:
	python case_studies/point_cloud_consistency/run_benchmark.py --device $(DEVICE) --output_dir results/point_cloud_consistency_train512 --train_points 512 --point_counts 64 128 256 512 1024 --reference_points 4096 --n_resamples 3

point-cloud-consistency-pointnext:
	python case_studies/point_cloud_consistency/run_benchmark.py --device $(DEVICE) --backbone pointnext --output_dir results/point_cloud_consistency_pointnext_run

point-cloud-consistency-512-pointnext:
	python case_studies/point_cloud_consistency/run_benchmark.py --device $(DEVICE) --backbone pointnext --output_dir results/point_cloud_consistency_pointnext_train512 --train_points 512 --point_counts 64 128 256 512 1024 --reference_points 4096 --n_resamples 3

point-cloud-mean-regression:
	python case_studies/point_cloud_consistency/run_mean_regression.py --device $(DEVICE) --output_dir results/point_cloud_mean_regression_run

point-cloud-mean-regression-pointnext:
	python case_studies/point_cloud_consistency/run_mean_regression.py --device $(DEVICE) --backbone pointnext --output_dir results/point_cloud_mean_regression_pointnext_run

sphere-reconstruction:
	python case_studies/sphere_signal_reconstruction/run_benchmark.py --device $(DEVICE) --output_dir results/sphere_signal_reconstruction_run

ahmedml-prepare:
	python case_studies/ahmedml_surface_forces/prepare_dataset.py --raw_root $(AHMEDML_RAW_ROOT) --output_dir $(AHMEDML_ROOT)

ahmedml-surface-forces:
	python case_studies/ahmedml_surface_forces/run_benchmark.py --device $(DEVICE) --processed_root $(AHMEDML_ROOT) --steps $(AHMEDML_STEPS) --output_dir results/ahmedml_surface_forces_run

ahmedml-surface-forces-pointnext:
	python case_studies/ahmedml_surface_forces/run_benchmark.py --device $(DEVICE) --processed_root $(AHMEDML_ROOT) --steps $(AHMEDML_STEPS) --output_dir results/ahmedml_surface_forces_pointnext_run --backbone pointnext

airfrans-prepare:
	python case_studies/airfrans_field_prediction/prepare_dataset.py --raw_root $(AIRFRANS_RAW_ROOT) --output_dir $(AIRFRANS_ROOT)

airfrans-force-regression:
	python case_studies/airfrans_field_prediction/run_benchmark.py --device $(DEVICE) --processed_root $(AIRFRANS_ROOT) --output_dir results/airfrans_field_prediction_run

airfrans-force-regression-pointnext:
	python case_studies/airfrans_field_prediction/run_benchmark.py --device $(DEVICE) --processed_root $(AIRFRANS_ROOT) --output_dir results/airfrans_field_prediction_pointnext_run --backbone pointnext

airfrans-field-prediction: airfrans-force-regression

airfrans-field-prediction-pointnext: airfrans-force-regression-pointnext

paper-assets:
	python case_studies/point_cloud_consistency/plot_regression_convergence.py --metrics_path results/point_cloud_mean_regression_run/metrics.json --output_dir results/point_cloud_mean_regression_run --pointnext_metrics_path results/point_cloud_mean_regression_pointnext_run/metrics.json
	python case_studies/point_cloud_consistency/plot_regression_qualitative.py --metrics_path results/point_cloud_mean_regression_run/metrics.json --output_dir results/point_cloud_mean_regression_run --pointnext_metrics_path results/point_cloud_mean_regression_pointnext_run/metrics.json
	python case_studies/ahmedml_surface_forces/plot_convergence.py --metrics_path results/ahmedml_surface_forces_run/metrics.json --output_dir results/ahmedml_surface_forces_run --pointnext_metrics_path results/ahmedml_surface_forces_pointnext_run/metrics.json
	python case_studies/ahmedml_surface_forces/plot_qualitative.py --metrics_path results/ahmedml_surface_forces_run/metrics.json --output_dir results/ahmedml_surface_forces_run --pointnext_metrics_path results/ahmedml_surface_forces_pointnext_run/metrics.json
	python case_studies/airfrans_field_prediction/plot_convergence.py --metrics_path results/airfrans_field_prediction_run/metrics.json --output_dir results/airfrans_field_prediction_run --pointnext_metrics_path results/airfrans_field_prediction_pointnext_run/metrics.json
	python case_studies/airfrans_field_prediction/plot_qualitative.py --metrics_path results/airfrans_field_prediction_run/metrics.json --output_dir results/airfrans_field_prediction_run --pointnext_metrics_path results/airfrans_field_prediction_pointnext_run/metrics.json

paper: paper-assets
	python paper/generate_result_tables.py
	tectonic paper/Set-Encoder.tex

clean:
	rm -rf results/ logs/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
