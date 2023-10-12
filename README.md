## Deps-SAN: Neural Machine Translation with Dependency-Scaled Self-Attention Network [[Paper]](https://arxiv.org/abs/2111.11707)
![](https://github.com/pengr/Deps-SAN/blob/master/Deps-SAN.png)


## Step1: Requirements
- Build running environment (two ways)
```shell
  1. pip install --editable .  
  2. python setup.py build_ext --inplace
````
- pytorch==1.7.0, torchvision==0.8.0, cudatoolkit=10.1 (pip install is also work)
```shell
  conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.1 -c pytorch 
````
- Python 3.7.6


## Step2: Data Preparation
The vanilla dataset used in this work is IWSLT'14 German to English dataset, it can be found at [here](http://workshop2014.iwslt.org/downloads/proceeding.pdf).
You can use the following script to preprocess raw text:
```bash
bash expriments/prepare-iwslt14-sdsa.sh
```

## Step3: Running code
You can let this code works by run the scripts in the directory *expriments*.

1. preprocess dataset into torch type
    ```bash
    bash pre_sdsa.sh
    ```
    
2. train model
    ```bash
    bash train_sdsa.sh
    ```
   
3. generate target sentence
    ```bash
    bash gen_sdsa.sh
    ```
4. RS-Sparsing and Wink-Sparsing variants can be run by appending the suffix *_rs* or *_wink*
    ```bash
    bash train_sdsa_rs.sh train_sdsa_wink.sh gen_sdsa_rs.sh gen_sdsa_wink.sh
    ```

## Citation
If you use the code in your research, please cite:
```bibtex
@inproceedings{peng2022deps,
    title={Deps-SAN: Neural Machine Translation with Dependency-Scaled Self-Attention Network},
    author={Peng, Ru and Lin, Nankai and Fang, Yi and Jiang, Shengyi and Hao, Tianyong and Chen, Boyu and Zhao, Junbo},
    booktitle={International Conference on Neural Information Processing},
    pages={26--37},
    year={2022},
    organization={Springer}
}
```
