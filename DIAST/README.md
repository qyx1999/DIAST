## Environment
- [PyTorch >= 1.7](https://pytorch.org/) **(Recommend **NOT** using torch 1.8!!! It would cause abnormal performance.)**
- [BasicSR == 1.3.4.9](https://github.com/XPixelGroup/BasicSR/blob/master/INSTALL.md) 

## How To Test

- Refer to `./options/test` for the configuration file of the model to be tested, and prepare the testing data and pretrained model.  
```
python net/test.py -opt options/test/SRx4.yml
```

## How To Train
- Refer to `./options/train` for the configuration file of the model to train.
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 net/train.py -opt options/train/SRx2.yml --launcher pytorch
```
- the default batch size per gpu is 4, which will cost about 20G memory for each GPU.  

