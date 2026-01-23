
# Experiment Results

## File Structure


The experiment results are organized in the following file structure: 

- `metrics_<version>.csv`: Contains raw metrics for each experiment run, including model, dataset, run, accuracy and loss of train, test and validation, training time, size in MBytes, and other parameters.
- `aggregated_by_dataset_metrics_<version>.csv`: Contains aggregated metrics for each dataset and dataset after filtering outliers, calculating the mean, std, and a 95% confidence interval.
- `summary_dataset_results.csv`: A summary of the aggregated results for each model and dataset, including mean accuracy with the confidence intervals.
- `summary_model_metrics_<version>.csv`: A summary of the aggregated results for each model across all datasets, including mean accuracy without confidence intervals and the mean ranking.


## About the Experiments

For more details on the experiments, please refer to the [README](../../src/experiments/README.md) file in the experiments' directory.
