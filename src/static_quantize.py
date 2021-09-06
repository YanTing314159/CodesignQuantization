import torch
import numpy as np
from data import CIFAR100
from utils import transform, evaluate_, get_model
import time
import datetime

np.random.seed(0)

def static_quantize(model_fp32, dataset, sample_size):
  assert sample_size <= len(dataset)
  quantize_samples = []
  rand = np.random.choice(len(dataset), sample_size)
  for i in rand:
   quantize_samples.append(dataset.__getitem__(i)[1].numpy())
  quantize_samples = np.array(quantize_samples)
  quantize_samples = torch.from_numpy(quantize_samples)

  model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
  model_fp32_prepared = torch.quantization.prepare(model_fp32)
  model_fp32_prepared(quantize_samples)
  model_int8 = torch.quantization.convert(model_fp32_prepared)

  return model_int8

#"""
model = get_model("resnet56_q")
model.load_state_dict(torch.load("../pth/cifar100_resnet56-f2eff4c8.pt"))
#"""
"""
model = get_model("vgg16_q") #acc:0.7373
model.load_state_dict(torch.load("../pth/cifar100_vgg16_bn-7d8c4031.pt"))
"""

dataset = CIFAR100("../cifar-100-python/", mode='train', transform=transform)
model_int8 = static_quantize(model, dataset, sample_size=100)
print("quantized!")
dataset_test = CIFAR100("../cifar-100-python/", mode='test', transform=transform)
evaluate_(model_int8, dataset_test)



