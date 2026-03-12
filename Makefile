.PHONY: reproduce ablation convergence weight-analysis point-cloud-consistency point-cloud-consistency-512 point-cloud-mean-regression sphere-reconstruction

DEVICE ?= cuda:0

reproduce: ablation convergence weight-analysis

ablation:
	python case_studies/darcy_1d/sensor_ablation.py --device $(DEVICE) --output_dir results

convergence:
	python case_studies/darcy_1d/plot_convergence.py --device $(DEVICE) --output_dir results --n_test 200

weight-analysis:
	python case_studies/darcy_1d/plot_weight_analysis.py --device $(DEVICE) --output_dir results --n_test 200 --sample_idx 5

point-cloud-consistency:
	python case_studies/point_cloud_consistency/run_benchmark.py --device $(DEVICE) --output_dir results/point_cloud_consistency_run

point-cloud-consistency-512:
	python case_studies/point_cloud_consistency/run_benchmark.py --device $(DEVICE) --output_dir results/point_cloud_consistency_train512 --train_points 512 --point_counts 64 128 256 512 1024 --reference_points 4096 --n_resamples 3

point-cloud-mean-regression:
	python case_studies/point_cloud_consistency/run_mean_regression.py --device $(DEVICE) --output_dir results/point_cloud_mean_regression_run

sphere-reconstruction:
	python case_studies/sphere_signal_reconstruction/run_benchmark.py --device $(DEVICE) --output_dir results/sphere_signal_reconstruction_run
