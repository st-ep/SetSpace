.PHONY: help point-cloud-consistency point-cloud-consistency-512 point-cloud-mean-regression sphere-reconstruction install clean

DEVICE ?= cuda:0

help:
	@echo "Usage: make <target> [DEVICE=cuda:0]"
	@echo ""
	@echo "Targets:"
	@echo "  install                      Install package in editable mode"
	@echo "  point-cloud-consistency      Synthetic point-cloud consistency benchmark"
	@echo "  point-cloud-consistency-512  Point-cloud benchmark (train at 512 points)"
	@echo "  point-cloud-mean-regression  Point-cloud mean-regression benchmark"
	@echo "  sphere-reconstruction        Sphere signal reconstruction benchmark"
	@echo "  clean                        Remove results, logs, and __pycache__"

install:
	pip install -e .

point-cloud-consistency:
	python case_studies/point_cloud_consistency/run_benchmark.py --device $(DEVICE) --output_dir results/point_cloud_consistency_run

point-cloud-consistency-512:
	python case_studies/point_cloud_consistency/run_benchmark.py --device $(DEVICE) --output_dir results/point_cloud_consistency_train512 --train_points 512 --point_counts 64 128 256 512 1024 --reference_points 4096 --n_resamples 3

point-cloud-mean-regression:
	python case_studies/point_cloud_consistency/run_mean_regression.py --device $(DEVICE) --output_dir results/point_cloud_mean_regression_run

sphere-reconstruction:
	python case_studies/sphere_signal_reconstruction/run_benchmark.py --device $(DEVICE) --output_dir results/sphere_signal_reconstruction_run

clean:
	rm -rf results/ logs/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
