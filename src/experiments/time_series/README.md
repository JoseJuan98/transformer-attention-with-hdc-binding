
# Time Series Classification Experiments

This directory contains the code for the experiments conducted for time series classification.

The experiments were conducted with the following parameters in the `epxeriment_cfg.json` file used for each experiment:

- `"num_epochs": 100`: number of epochs
- `"learning_rate": 0.003`: learning rate
- `"d_model": 64`: dimension of the model
- `"num_heads": 8`: number of attention heads
- `"d_ff": 256`: dimension of the feedforward network (x4 expansion factor)
- `"num_layers": 4`: number of encoder layers
- `"dropout": 0.1`: dropout rate
- `"num_layers": 1`: number of encoder layers ($N_L$). For all tests, the number of encoder layers is set to 1, except for the experiments with $N_L=4$.

The paremeters above follows the parameters used for the experiments carried out in the ConvTran paper (Foumani et al., 2024) [2].

Additionally, the experiments were conducted following effective training techniques, represented by the following parameters:

- `"runs_per_experiment": 10`: each model is trained 10 times with different random seeds for each experiment and dataset.
This is to ensure that the results are not biased by a single run. Additionally, when calculating the metrics a confidence interval of 95% is used.
- `"default_batch_size": 64`: default batch size. If `auto_scale_batch_size` is set to true, the batch size will be automatically scaled based on the available GPU memory,
otherwise, the default batch size will be used. Also, it's used to calculate the minimum gradient accumulation steps.
- `"accelerator": "auto"`: used by PyTorch Lightning to automatically select the best accelerator (GPU or CPU) based on the available resources.
- `"precision": "16-mixed"`: mixed 16 bit precision training. This is used to reduce the memory usage and speed up the training process.
- `"auto_scale_batch_size": true`: if set to true, the batch size will be automatically scaled based on the available GPU memory.
- `"metrics_mode": "append"`: if set to `append`, the metrics will be appended to the existing metrics file. If set to `write`, the existing metrics file will be overwritten.
Useful when restarting the training process.
- `"profiler": false`: if set to true, the profiler will be enabled. This is used to profile the training process and identify bottlenecks.
- `"summary": false`: if set to true, the summary will be printed at the end of the training process. This is used to print the model summary and the number of parameters.
- `"plots": true`: if set to true, the plots will be generated at the end of the training process. This is used to generate the plots for the training process.
- `"development": false`: if set to true, the training process will be run in development mode. This is used to run the training process with a small dataset, 2 runs and 2 epochs per model for debugging purposes.
- `"validation_split": 0.2`: percentage of the dataset to be used for validation. The rest of the dataset will be used for training.
- `"early_stopping_patience": 10`: number of epochs to wait for the validation loss to improve before stopping the training process. This is used to prevent overfitting.
- `"accumulate_grad_batches": 1`: number of batches to accumulate gradients before updating the model parameters. This is used to reduce the memory usage and speed up the training process.
- `"gradient_clip_val": 1`: value to clip the gradients. This is used to prevent exploding gradients.
- `"gradient_clip_algorithm": "norm"`: algorithm to use for clipping the gradients. This is used to prevent exploding gradients.
- `"use_swa": false`: if set to true, Stochastic Weight Averaging (SWA) will be used. This is used to improve the generalization of the model.
- `"swa_learning_rate": 0.0005`: learning rate for SWA. This is used to improve the generalization of the model.
- `"use_lr_finder": true`: if set to true, the learning rate finder will be used. This is used to find the optimal learning rate for the model.
- `"lr_finder_milestones": [0, 10, 20, 30, 50, 75]`: milestones for the learning rate finder. This is used to find the optimal learning rate for the model.

The datasets used for experiments and automatically downloaded are:

```
[
        "FaceDetection",
        "InsectWingbeat",
        "PenDigits",
        "SpokenArabicDigits",
        "LSST",
        "FingerMovements",
        "MotorImagery",
        "SelfRegulationSCP1",
        "Heartbeat",
        "SelfRegulationSCP2",
        "PhonemeSpectra",
        "CharacterTrajectories",
        "EthanolConcentration",
        "HandMovementDirection",
        "PEMS-SF",
        "RacketSports",
        "Epilepsy",
        "JapaneseVowels",
        "NATOPS",
        "EigenWorms",
        "UWaveGestureLibraryAll",
        "Libras",
        "ArticularyWordRecognition",
        "BasicMotions",
        "DuckDuckGeese",
        "Cricket",
        "Handwriting",
        "ERing",
        "AtrialFibrillation",
        "StandWalkJump"
]
```

## 1. Binding Method Experiments

Experiments with different binding methods, embedding methods and Absolute Sinusoidal Positional Encodings.

Binding methods evaluated:

- `additive` (Vaswani et al., 2017) [1]
- `multiplicative` (Component-wise multiplication)
- `convolutional` (Circular convolution)

Also, two kind of embeddings methods were evaluated with each binding method:

- Linear Projection
- 1D Convolution Feature Extraction

These results in the following combinations:
  - Additive Binding with Linear Projection Embeddings
  - Additive Binding with 1D Convolution Embeddings
  - Component-wise Binding with Linear Projection Embeddings
  - Component-wise Binding with 1D Convolution Embeddings
  - Circular Convolution Binding with Linear Projection Embeddings
  - Circular Convolution Binding with 1D Convolution Embeddings

### 1.1 - $N_L=4$

Run with $N_L=4$ (4 encoder layers) a part from the base parameters, the metrics in the folder [version_2](../../../docs/experiment_metrics/binding_methods/version_2)

#### 1.2 - $N_L=1$

Run with $N_L=1$ (1 encoder layers) a part from the base parameters, the metrics in the folder [version_3](../../../docs/experiment_metrics/binding_methods/version_3)

## 2. Positional Encoding Experiments

### 2.1 - Fractional Power Encoding and Learnable Encoding Experiments

- `positional_encodings`: experiments with different positional encodings with the best combination of binding method and
embedding method, Circular Convolution Binding with Linear Projection Embeddings. They include:
[//]: # (  - Isolated Sinusoidal Positional Encodings #TODO)
  - No Positional Encodings
  - Random Positional Encodings
  - Learned Positional Encodings based on a Random Normal Distribution
  - Absolute Sinusoidal Positional Encodings
  - Learned Positional Encodings based on the Absolute Sinusoidal Positional Encodings
  - Fractional Power Positional Encodings based in a Gaussian kernel with $\beta=0.8$
  - Fractional Power Positional Encodings based in a Sinc function with $\beta=1$
  - Fractional Power Positional Encodings based in a Sinc function with $\beta=2$
  - Fractional Power Positional Encodings based in a Sinc function with $\beta=5$

The results are in the folder [version_4](../../../docs/experiment_metrics/positional_encoding/pe_version_1)
