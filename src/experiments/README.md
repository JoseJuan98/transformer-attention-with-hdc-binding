
# Experiments

This directory contains the code for the experiments conducted as part of my Master's thesis
[Enhancing Transformer Attention with HDC Binding for Positional Encodings]()
at [Luleå University of Technology](https://www.ltu.se/en).

Due to the lack of computational resources, the experiments were for time series classification adapting the original Transformer architecture (Vaswani et al., 2017) [1].
To learn more about the architecture, please refer to the [README](../../README.md) file in the root directory.

Additionally, the modular implementation makes it easy to experiment with different types of embedding methods, binding methods, and positional encodings.

The experiments are organized into different folders based on the type of experiment. The experiments include:

- `time_series`: experiments for time series classification
  - `binding_methods`: experiments with different binding methods and embedding methods
  - `positional_encodings`: experiments with different positional encodings
- `positional_encodings`: visualizations of positional encodings
  - `ts_pe_visualization.py`: visualizes the positional encodings for time series data
  - `similarity_visualization.py`: visualizes the similarity between different positional encodings
