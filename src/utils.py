import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from model.resnet import CifarResNet, BasicBlock, CifarResNet_q, BasicBlock_q
from model.vgg import VGG, make_layers, VGG_q

def transform(img):
  mean = [0.5070, 0.4865, 0.4409]
  std = [0.2673, 0.2564, 0.2761]
  img = img.astype(np.float32)
  img = img / 255
  img = (img - mean)/std
  img = img.astype(np.float32)
  img = torch.from_numpy(img)
  img = img.permute(2, 0, 1)
  return img

def evaluate_(model, dataset, device='cpu'):
  dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
  evaluate(model, dataloader, device)

def evaluate(model, dataloader, device='cpu'):
  tot = 0
  cor = 0
  model.to(device)
  model.eval()
  st = time.time()
  for i, (label, image) in enumerate(dataloader):
    #image = image.unsqueeze(0)
    image = image.to(device)
    pred = model(image)
    if device == 'cuda':
      pred_label = np.argmax(pred.detach().cpu().numpy(), axis=1)
    else:
      pred_label = np.argmax(pred.detach().numpy(), axis=1)
    #print(pred_label, label)
    tot += pred_label.shape[0]
    cor += np.sum(pred_label == label.numpy())
    #break
    if i % 100 == 0:
      print(f"processing:{i}/{len(dataloader)}")
  print(f"acc:{cor/tot:.4f}, evaluate time:{time.time()-st:4f} sec")

def get_model(name):
  if name == 'resnet56':
    layers = [9]*3
    model = CifarResNet(BasicBlock, layers, num_classes=100)
    return model
  elif name == 'resnet56_q':
    layers = [9]*3
    model = CifarResNet_q(BasicBlock_q, layers, num_classes=100)
    return model
  elif name == 'vgg16':
    vgg16_cnfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    model = VGG(make_layers(vgg16_cnfg, batch_norm=True), num_classes=100, init_weights=False)
    return model
  elif name == 'vgg16_q':
    vgg16_cnfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    model = VGG_q(make_layers(vgg16_cnfg, batch_norm=True), num_classes=100, init_weights=False)
    return model
  else:
    print("Unsupport model!")
    return None

def print_key(model, num=5):
  _ = 0
  keys = []
  for k in model.state_dict():
    keys.append(f"key{_+1}: " + k)
    _ += 1
    if _ >= num:
      break
  print(keys)
