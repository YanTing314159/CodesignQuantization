import torch
from torch.utils.data import DataLoader
import numpy as np
from data import CIFAR100
from utils import transform, evaluate_, evaluate, get_model
from utils import print_key
import time

"""#pretrained resnet56 on cifar100, acc:0.7263
model = get_model("resnet56")
model.load_state_dict(torch.load("../pth/cifar100_resnet56-f2eff4c8.pt"))
dataset_test = CIFAR100("../cifar-100-python/", mode='test', transform=transform)
evaluate_(model, dataset_test, device='cuda')
"""

#"""#quantized resnet56, QAT, acc:0.7216
model = get_model("resnet56_q")
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)
model.load_state_dict(torch.load("../pth/resnet56_tuned20.pth"))
q_model = torch.quantization.convert(model.eval(), inplace=False)
q_model.load_state_dict(torch.load("../pth/resnet56_quantized20.pth"))
testset = CIFAR100("../cifar-100-python/", mode='test', transform=transform)
test_loader = DataLoader(testset, batch_size=128, shuffle=False)
evaluate(model, test_loader)
#"""

"""#pretrained VGG16 on cifar100, acc:74.00
model = get_model("vgg16")
model.load_state_dict(torch.load("../pth/cifar100_vgg16_bn-7d8c4031.pt"))
dataset_test = CIFAR100("../cifar-100-python/", mode='test', transform=transform)
evaluate_(model, dataset_test, device='cpu')
"""

"""#quantized VGG16, QAT, acc:0.7386
model = get_model("vgg16_q")
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)
model.load_state_dict(torch.load("../pth/vgg16_tuned20.pth"))
q_model = torch.quantization.convert(model.eval(), inplace=False)
q_model.load_state_dict(torch.load("../pth/vgg16_quantized20.pth"))
#torch.jit.save(torch.jit.script(q_model), "../pth/vgg16_quantized_fullmodel.mod")
testset = CIFAR100("../cifar-100-python/", mode='test', transform=transform)
test_loader = DataLoader(testset, batch_size=128, shuffle=False)
evaluate(q_model, test_loader)
"""

"""
q_model = torch.jit.load("../pth/vgg16_quantized.mod")
testset = CIFAR100("../cifar-100-python/", mode='test', transform=transform)
test_loader = DataLoader(testset, batch_size=128, shuffle=False)
#evaluate(q_model, test_loader)

model = get_model("vgg16_q")
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)
q2_model = torch.quantization.convert(model.eval(), inplace=False)
print_key(q_model, 1)
print_key(q2_model, 1)
q2_model.load_state_dict(q_model.state_dict(), strict=False)
evaluate(q2_model, test_loader)
"""

