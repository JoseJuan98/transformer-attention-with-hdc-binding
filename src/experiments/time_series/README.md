
# Time Series Classification Experiments

- `binding_methods`: experiments with different binding methods, embedding methods and Absolute Sinusoidal Positional
Encodings, including:
  - Additive Binding with Linear Projection Embeddings
  - Additive Binding with 1D Convolution Embeddings
  - Component-wise Binding with Linear Projection Embeddings
  - Component-wise Binding with 1D Convolution Embeddings
  - Circular Convolution Binding with Linear Projection Embeddings
  - Circular Convolution Binding with 1D Convolution Embeddings

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
