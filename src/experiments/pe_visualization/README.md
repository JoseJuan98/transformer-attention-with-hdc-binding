
# Positional Encoding Experiments


## Positional Encoding Visualization

These plots visualize the positional encodings values and their similarities. The goal is to understand how different positional encodings behave and how they can be used in the context of time series classification.
The results are in the [positional_encodings](../../../docs/plots/positional_encodings) directory with their respective name as subfolder, excluding the
`similarity` subfolder, which contains the similarity plots.

Explanation of the plots generated by the [ts_pe_visualization.py](positional_encoding.py) script for positional encodings (PE):

- Heatmap of the values: This gives an overall view.
  - Isolated Sinusoidal PE: there are two halves (sine part and cosine part) because of the split between cosines and sines ("Isolated"). Also,
  it's observed vertical bands for the low dimensions (slowly changing frequencies across positions) and more horizontal-like,
  rapidly changing patterns for the higher dimensions (high frequencies).
- Specific Position Vectors: These plots helps visualize what the actual vector looks like for a given time step.
  - Isolated Sinusoidal PE: there are wave patterns across the embedding dimension. Comparing lines for different positions
  shows how the encoding uniquely identifies each position.
- Specific Dimension Values: These plots highlights the core idea of sinusoidal encoding: using waves of different
frequencies. Lines for low dimensions (e.g., 0, 1) will look like slow sine/cosine waves, while lines for high dimensions
(e.g., 126, 127) will oscillate much faster as you move along the sequence positions. This confirms that different dimensions
capture positional information at different granularities.

## Positional Encoding Relative Similarity

These plots visualize the similarity between different positions for different positional encodings. The goal is to understand how similar the positional encodings are for different positions and how they can be used in the context of time series classification.
The results are in the [similarity](../../../docs/plots/positional_encodings/similarity) directory with their respective name as subfolder.

Explanation of the plots generated by the [smilarity_visualization.py](similarity_shapes.py) script for positional encodings (PE):
- Cosine Similarity relative to the center position of all the positional encodings: This gives an overall view of the similarity between the positional encodings.
- Fractional Power Encodings Cosine Similarity with different kernels and $\beta$ values: This shows how the similarity changes with different kernels and $\beta$ values.
- Cosine Similarity relative to position 0 for the Sinusoidal PE and the Fractional Power Encodings with different kernels and $\beta$ values: This shows how the similarity changes with different kernels and $\beta$ values.
- Cosine Similarity relative to the position in the first quarter of the sequence  for the Sinusoidal PE and the Fractional Power Encodings with different kernels and $\beta$ values: This shows how the similarity changes with different kernels and $\beta$ values.
- Cosine Similarity relative to the position in the center of the sequence  for the Sinusoidal PE and the Fractional Power Encodings with different kernels and $\beta$ values: This shows how the similarity changes with different kernels and $\beta$ values.
- Cosine Similarity relative to the position in the last quarter of the sequence  for the Sinusoidal PE and the Fractional Power Encodings with different kernels and $\beta$ values: This shows how the similarity changes with different kernels and $\beta$ values.
- Cosine Similarity relative to the position in the last position of the sequence  for the Sinusoidal PE and the Fractional Power Encodings with different kernels and $\beta$ values: This shows how the similarity changes with different kernels and $\beta$ values.
- Product Similarity relative to the center position of all the positional encodings.
