# OIGAN

PyTorch implementation of **"Extended depth-of-field resolution enhancement microscopy imaging for neutralizing the impact of mineral inhomogeneous surface"**

## Dependencies
- Python 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch = 1.10.0](https://pytorch.org/)

## Preparation
The **pretrained models** can be find in Google Drive **"[/save_model](https://drive.google.com/drive/folders/11kofSwlP8lfJP2aZAXmZoDPVe5qsJAtW?usp=sharing)"**. Two models based on plane polarized light mode and cross polarized light mode respectively are available.

## Parameters
The configurations in `option/test/`can be modified. We provide the test parameters of models suitable for plane polarized light and cross polarized light images. Here are the details:

```c++
{
  "name": "OIGAN_cross"
  , "model": "oigan"
  , "scale": 4
  , "gpu_ids": []
  ,"pretrain": 1
  ,"datasets": {
    "test": {
      "name": "dataset_B"
      , "mode": "LRHR" // or LR
      , "dataroot_HR": "data/HR/dataset_B/" // If in "LR" mode, delect it
      , "dataroot_LR": "data/LR/dataset_B/"
      , "subset_file": null
      , "use_shuffle": true
      , "n_workers": 1
      , "batch_size": 1
      , "HR_size": 512
      , "use_flip": false
      , "use_rot": false
    }
  }
  , "path": {
    "pretrain_dir":  "save_model/cross/"
    ,"pretrain_model_G": "save_model/cross/XXXX_G.pth"
    ,"pretrain_now": "XXXX_G.pth"
    ,"results_save_root": "result/" 
  }
  , "network_G": {
    "norm_type": null
    , "mode": "CNA"
    , "nf": 64
    , "nb": 23
    , "in_nc": 3
    , "out_nc": 3
    , "gc": 32
    , "group": 1
  }
  ,"is_train": false
  ,"save_pic": true //wether save the reconstruction result
  ,"save_pic_excel": true // wether save the metrics in tabular form
  ,"zb_order": ["name","ssim","psnr","lpips"] // metrics used

}
```
    


## Test Example

Let's create an example. Before running this code, please modify option files to your own configurations including: 
  - proper `test-mode` and `data_LR/HR` paths for the data loader. Set `mode` to `LRHR` if you can provide label data and expect to test the resulting metrics; otherwise, if only reconstruction is performed, choose `LR` mode.
  - proper `result_save_root` for the test results.
  - control what you want to save in the test process through `save_pic`, `save_pic_excal` and `zb_order`.

To Now, you can implement the super resolution reconstruction of microscopic images of minerals with the following code:
```
python test_model.py
```
The test results of each model will be saved in path "`result_save_root`/`name`/`pretrain_now`", and the test logs are synchronously saved in path "`result_save_root`/`name`/test.log"
## Results
### Quantitative Results
Deep-learning based methods were compared to demonstrate the power of the proposed model. 

PSNR/LPIPS comparison with ESRGAN, USISGAN, DUSGAN and Beby-GAN.

| Method | Dataset A | Dataset B | Dataset C | Dataset D | 
| --------------------------------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- |  
| USISGAN | 18\.93/0\.663 | 18\.65/0\.653 | 18\.75/0\.759 | 17\.41/0\.702 | 
| ESRGAN | 18\.28/0\.444 | 18\.30/0\.441 | 17\.74/0\.474 | 17\.98/0\.464 |
| DUGAN | 18\.85/0\.604 | 18\.60/0\.606 | 18\.72/0\.672 | 17\.38/0\.670 | 
| Beby-GAN | 19\.10/0\.416 | 19\.15/0\.416 | 19\.44/0\.498 | 20\.53/0\.508 |
| OURS | **19\.25**/**0\.413** | **19\.19**/**0\.378** | **19\.46**/**0\.403** | **22\.07**/**0\.434** | 


## Acknowledgement
The code is based on [BasicSR](https://github.com/xinntao/BasicSR), [SPSR](https://github.com/Maclory/SPSR) and [LPIPS](https://github.com/richzhang/PerceptualSimilarity). 

