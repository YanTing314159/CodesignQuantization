import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from data import CIFAR100
from utils import transform, evaluate, get_model
import time
import datetime

np.random.seed(0)

def QAT(model, criterion, optimizer, data_loader, device, train_batches, test_loader=None):
  st = time.time()
  step = 0
  model.train()
  for batch in range(train_batches):
    print(f"start batch {batch}")
    model.to(device)
    model.train()
    torch.manual_seed(batch)
    for i, (label, image) in enumerate(data_loader):
      label, image = label.to(device), image.to(device)
      output = model(image)
      loss = criterion(output, label)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      step += 1
      if (step+1) % 2 == 0:
        print(f"training batch {batch+1}, step {step+1}, loss={loss.item()}")
        break
    model.to("cpu")
    q_model = torch.quantization.convert(model.eval(), inplace=False)
    """
    if test_loader:
      evaluate(q_model, test_loader)
    else:
      evaluate(q_model, data_loader)
    """
  return model, q_model
#"""
model = get_model("resnet56_q")
model.load_state_dict(torch.load("../pth/cifar100_resnet56-f2eff4c8.pt"))
#"""
"""
model = get_model("vgg16_q")
model.load_state_dict(torch.load("../pth/cifar100_vgg16_bn-7d8c4031.pt"))
"""
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)
dataset = CIFAR100("../cifar-100-python/", mode='train', transform=transform)
testset = CIFAR100("../cifar-100-python/", mode='test', transform=transform)
data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(testset, batch_size=128, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
device = 'cpu'
model.to(device)

model, q_model = QAT(model, criterion, optimizer, data_loader, device, 20, test_loader)
"""
torch.save(model.state_dict(), "./vgg16_tuned20.pth")
torch.save(q_model.state_dict(), "./vgg16_quantized20.pth")
evaluate(q_model, test_loader)
"""

