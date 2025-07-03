## PFDBooster
This is the offical implementation for the paper titled "PFDBooster: A Unified Post-Image Fusion Dual-Domain Boosting Paradigm".


## <img width="40" src="Figs/environment.png"> Environment
```
requirements.txt
```

## <img width="15" src="Figs/dataset.png"> Dataset:LLVIP

[Training Set (DDcGAN Results on LLVIP)](https://pan.baidu.com/s/1X58UeWpLSBiFMlRi6pFOLw?pwd=hokf) Password: hokf

[Training Set (Original LLVIP)](https://pan.baidu.com/s/1_I707esOlERfyMiUOzuZQg?pwd=jq15) Password: jq15

Put the above train data in the "train_data" folder


## <img width="32" src="Figs/train.png"> Train

### <img width="20" src="Figs/task.png"> IVIF task (Backbone: DDcGAN)

```
python train.py
```

The trained model will be saved in the "models" folder automatically.


## <img width="32" src="Figs/test.png"> Test 

### <img width="25" src="Figs/set.png"> Fusion Images (Booster Only)

To use our pre-trained PFDBooster to boost an arbitary method:

<img width="20" src="Figs/task.png"> IVIF task (Backbone: DDcGAN)

```
python test_booster_only_IVIF_rgb.py
```
or 

```
python test_booster_only_IVIF_gray.py
```
You can modify the path in the "test_booster_only_xxxx.py" file, to enhance your own fusion results. 


### <img width="25" src="Figs/set.png"> IR or VIS Images (End to end)

Use our pre-trained model to directly output enhanced fusion results based on two input images.

<img width="20" src="Figs/task.png"> IVIF task (Backbone: MUFusion):

```
python test_e2e_IVIF_rgb.py
```

<img width="20" src="Figs/task.png"> MFIF task (Backbone: MUFusion):

```
python test_e2e_MFIF_rgb.py
```

## <img width="32" src="Figs/highlight.png"> Highlight
- We devise an image fusion booster by analysing the quality of the initial fusion results by means of a dedicated Information Probe.
- The proposed PFDBooster is a general enhancer, which can be applied to various image fusion methods, e.g., traditional or learning-based algorithms, irrespective of the type of fusion task.
- In a new divide-and-conquer image fusion paradigm, the results of the analysis performed by the Information Probe guide the refinement of the fused image.
- The proposed PFDBooster significantly enhances the performance of the SOTA fusion methods and downstream detection tasks, with only a slight increase in the computational overhead.

## <img width="32" src="Figs/citation.png"> Citation
If this work is helpful to you, please cite our paper.

