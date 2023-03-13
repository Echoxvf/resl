### Requirement

- numpy
- torch==1.8.1
- torchvision
- six
- h5py
- Pillow
- scipy
- scikit-learn
- metric-learn 
- metric-learn 
- faiss_gpu==1.6.3
- dropblock==0.3.0
- pyyaml
- yacs
- termcolor
- tabulate
- gdown

### datasers
you should prepare dataset like [ECN](https://github.com/zhunzhong07/ECN) /[JVTC](https://github.com/ljn114514/JVTC) and unzip each dataset and corresponding CamStyle under 'ECN/data/' as following:

<pre>
.
+-- examples/data
|   +-- Market-1501-v15.09.15
|           +-- bounding_box_train
|           +-- query
|           +-- bounding_box_test
|           +-- bounding_box_train_camstyle
|   +-- DukeMTMC-reID
|           +-- bounding_box_train
|           +-- query
|           +-- bounding_box_test
|           +-- bounding_box_train_camstyle
|   +-- MSMT17_V1
|           +-- bounding_box_train
|           +-- bounding_box_test
|           +-- bounding_box_train_camstyle
|           +-- query
+ -- other files in this repo

</pre>
You can download the dataset from: https://pan.baidu.com/s/1vwApZ0St6KBWMapMGq5ALw 提取码：o3mk 

### train
We run our models on one NVIDIA 3090 GPUs. 

1.Pretrained on source domain. 
Using the fast-reid to train ReSL model on the source domain dataset

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_net_unsupervised.py --config-file configs/Unsupervised/sbs_R50_resl_duke_supervised.yml 

CUDA_VISIBLE_DEVICES=0 python tools/train_net_unsupervised.py --config-file configs/Unsupervised/sbs_R50_resl_market_supervised.yml 
```


2.Training on target dataset

```python
CUDA_VISIBLE_DEVICES=0  python tools/train_net_unsupervised.py --config-file configs/Unsupervised/sbs_R50_resl_market.yml MODEL.WEIGHTS /path/to/checkpoint_file

CUDA_VISIBLE_DEVICES=0 python tools/train_net_unsupervised.py --config-file configs/Unsupervised/sbs_R50_resl_duke.yml  MODEL.WEIGHTS /path/to/checkpoint_file

CUDA_VISIBLE_DEVICES=0 python tools/train_net_unsupervised.py --config-file configs/Unsupervised/sbs_R50_resl_msmt17.yml  MODEL.WEIGHTS /path/to/checkpoint_file
```


3.Fully unsupervised training.


```python
CUDA_VISIBLE_DEVICES=0  python tools/train_net_unsupervised.py --config-file configs/Unsupervised/sbs_R50_resl_market.yml

CUDA_VISIBLE_DEVICES=0 python tools/train_net_unsupervised.py --config-file configs/Unsupervised/sbs_R50_resl_duke.yml 

CUDA_VISIBLE_DEVICES=0 python tools/train_net_unsupervised.py --config-file configs/Unsupervised/sbs_R50_resl_msmt17.yml
```