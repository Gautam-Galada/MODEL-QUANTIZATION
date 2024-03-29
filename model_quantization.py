

import os

!pip install transformers

import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

"""BASED ON PEGASUS, MODEL QUANTIZATION WORKS WELL WITH SEQ2SEQ MODEL AND IS HELPFUL FOR DEPLOYMENT PROCESS"""

model_ckpt = "google/pegasus-cnn_dailymail"
model = PegasusForConditionalGeneration.from_pretrained(model_ckpt)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

unquant_model = model.state_dict()
torch.save(unquant_model, "pegasus-unquant.h5")

model.state_dict().keys()

quantized_model.config.save_pretrained("pegasus-quantized-config")
quantized_state_dict = quantized_model.state_dict()
torch.save(quantized_state_dict, "pegasus-quantized.h5")

from transformers import AutoConfig

config = AutoConfig.from_pretrained("pegasus-quantized-config")
dummy_model = PegasusForConditionalGeneration(config)

reconstructed_quantized_model = torch.quantization.quantize_dynamic(
    dummy_model, {torch.nn.Linear}, dtype=torch.qint8
)
reconstructed_quantized_model.load_state_dict(quantized_state_dict)

#incase u save the model files locally
quantized_state_dict = torch.load("pegasus-quantized.h5")

quantized_model.load_state_dict(quantized_state_dict)

quantized_model.state_dict().keys()

quantized_model

if model.state_dict().keys() == quantized_model.state_dict().keys():
  print("INTERNAL CONFIGURATION OF LAYERS STAYS CONSTANT")

unquantized_model_size = os.path.getsize("/content/pegasus-unquant.h5")

quantized_model_size = os.path.getsize("/content/pegasus-quantized.h5")

from pathlib import Path

Path("/content/pegasus-quantized.h5").stat()

Path("/content/pegasus-unquant.h5").stat()

print("UNQUANTIZED MODEL SIZE :", unquantized_model_size/(1024*1024),  "MB")
print("QUANTIZED MODEL SIZE   :", quantized_model_size/(1024*1024), "MB")

if model.config == quantized_model.config:
  print("MODEL ARCHITECTURE IS CONSTANT")
