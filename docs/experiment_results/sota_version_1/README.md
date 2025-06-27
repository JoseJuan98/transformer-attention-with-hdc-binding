

[//]: # (# Notes)

[//]: # ()
[//]: # ()
[//]: # (Big datasets skipped because of limited resources:)

[//]: # (- "MotorImagery")

[//]: # (- "InsectWingbeat")

[//]: # ()
[//]: # (Errors:)

[//]: # (- SpokenArabicDigits | convtran | Run 1:)

[//]: # (```)

[//]: # (  File "transformer-attention-with-hdc-binding/src/models/transformer/attention/erpe_attention.py", line 139, in forward)

[//]: # (    attn_weights_with_bias = attention_weights + relative_biases)

[//]: # (                             ~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~)

[//]: # (RuntimeError: The size of tensor a &#40;83&#41; must match the size of tensor b &#40;93&#41; at non-singleton dimension 3)

[//]: # (```)

[//]: # ()
[//]: # (- PEMS-SF | convtran | Run 1:)

[//]: # (```)

[//]: # (Error for PEMS-SF training convtran in run 1:)

[//]: # ()
[//]: # (CUDA out of memory. Tried to allocate 1.06 GiB. GPU 0 has a total capacity of 3.81 GiB of which 336.56 MiB is free. Including non-PyTorch memory, this process has 3.47 GiB memory in use. Of the allocated memory 2.31 GiB is allocated by PyTorch, and 1.02 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  &#40;https://pytorch.org/docs/stable/notes/cuda.html#environment-variables&#41;)

[//]: # (```)
