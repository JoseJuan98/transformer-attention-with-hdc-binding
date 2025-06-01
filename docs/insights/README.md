
# Insights

## High Dimension Dataset Insights

### Face Detection

It seems a hard dataset to learn from, with 144 dimensions.

- `overfitting_1d_conv_circular_conv_isolated_sinusoidal_run_1_binding_version_1.png` and `overfitting_1d_conv_circular_conv_sinusoidal_run_1_binding_version_1.png` show clear signs of overfitting.

- `face_detection_linear_conv_learnable_sinusoidal_run_1_pe_version_1.png`: meanwhile in this case still overfitting is observed, the learnable PE seems to have a smoother validation loss, indicating better generalization, though not for many epochs.

- `feace_detection_linear_conv_sinusoidal_run_9_pe_version_1.png`: seems to generalize better
