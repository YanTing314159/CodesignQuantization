# CodesignQuantization

From pretrained FP32 model to quantized int8 model.
Pretrained FP32 model using CIFAR100 dataset.
Pretrained model reference: https://github.com/chenyaofo/pytorch-cifar-models

/
--pth/
  --cifar100_resnet56-f2eff4c8.pt #pretrained FP32 resnet56
  --cifar100_vgg16_bn-7d8c4031.pt #pretrained FP32 vgg16
  --resnet56_quantized20.pth #qunatized model form "resnet56_tuned20.pth"
  --resnet56_tuned20.pth #QAT resnet56, with default QAT config, train 20 epoch
  --vgg16_quantized20 #qunatized model form "vgg16_tuned20.pth"
  --vgg16_tuned20 #QAT vgg16, with default QAT config, train 20 epoch
--src/
  --model/
    --resnet.py #resnet definition
	--vgg.py #vgg definition
  --data.py #CIFAR100 dataset
  --QAT.py #QAT
  --static_quantize.py #static quantization
  --test.py #load model weight and evaluate
  --utils.py #some processing, evaluating, ... functions
--acc.txt #model accuracy log