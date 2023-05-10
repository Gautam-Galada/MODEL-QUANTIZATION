# Pegasus Model Quantization for Deployment

This code demonstrates how to quantize a Pegasus model using PyTorch's dynamic quantization feature. Model quantization is helpful for deployment purposes as it reduces the size of the model and improves inference speed. 

The code uses the following steps to quantize the Pegasus model:

1. Load the pre-trained Pegasus model from the Hugging Face model hub.
2. Quantize the model using PyTorch's `quantize_dynamic` function.
3. Save the unquantized model state dict as `pegasus-unquant.h5`.
4. Save the quantized model's configuration as `pegasus-quantized-config`.
5. Save the quantized model state dict as `pegasus-quantized.h5`.
6. Load the quantized model state dict and check if the internal configuration of layers stays constant.
7. Calculate the size of the unquantized and quantized models and print the sizes.

## Dependencies

The following dependencies need to be installed before running the code:

- transformers
- torch

## Usage

- Clone the repository
- Install the dependencies: `pip install -r requirements.txt`
- Run the `pegasus_quantization.ipynb` notebook

## Acknowledgements

This code is based on the Pegasus model from the Hugging Face model hub and PyTorch's dynamic quantization feature.
