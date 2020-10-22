# Pytorch Implementation of Multiscale Residual Networks for Image Super-Resolution

***

This repository is an __unofficial__ Pytorch implementation of the ECCV 2018 paper "Multiscale Residual Networks for Image Super-Resolution". [Super-resolution](https://en.wikipedia.org/wiki/Super-resolution_imaging) is the computer vision problem of increasing the resolution of a low-resolution image and has applications in quality improvement of images. The problem has been approached using both traditional and deep learning approaches. This is a __personal project__ which was implemented for the purpose of practicing Pytorch. For the official sources, visit the [GitHub page](https://github.com/MIVRC/MSRN-PyTorch) and the [published paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Juncheng_Li_Multi-scale_Residual_Network_ECCV_2018_paper.pdf).

### Requirements

* Python 3
* Numpy == 1.19.1
* OpenCV >= 4.1.0
* Pytorch >= 1.6.0
* Tensorboard == 2.2.1

### Data

The dataset used for training this network is [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/), which consists of 900 high res and low res image pairs. The ```Div2K``` class is defined as a subset of ```Dataset``` to load the DIV2K images at run-time using OpenCV.

For testing, the Set14 testing dataset is run on the model for 2x and 4x super-resolution. The download link can be found [here](https://cvnote.ddlee.cn/2019/09/22/image-super-resolution-datasets). The ```Set14``` class is defined as a subset of ```Dataset``` to load the images using OpenCV at run-time, and the process can be repeated for testing any other dataset.

### Training

The training script can be found in ```train.py```. The argument parser in the script lists a variety of training arguments, out of which, the required arguments are -

* ```--data_root``` - the root directory of the DIV2K dataset (leave the underlying directory structure unchanged)
* ```--scale``` - the super-resolution scale of the output, which can be 2, 3, or 4

The default settings, including the required arguments, are good enough to train the model as per the training parameters given in the paper. Other important arguments include -

 * ```--residual_blocks``` - number of residual blocks in the network (default is 8)
 * ```--use_cpu``` - boolean to indicate whether to train on CPU (default is False)
 * ```--patch_size``` - the patch-size of the input image that the network is trained on. The authors state that training it on a random patch of small size improves speed of training (default is 128)
 * ```--loss_fn``` - the loss function to use. May implement GAN loss in the future (default is L1)

 The training loop overrides a single checkpoint file every epoch in case the training is interrupted. In case of interruption, you can set the ```--resume``` parameter to ```True```, and specify a ```--checkpoint_file``` to resume training from that checkpoint. The code additionally stores the model every 100th epoch for prediction purposes.

An example command for training is -

```
python train.py --data_root DIV2K/ --scale 4 --results_dir training_results/
```

### Testing

Testing is currently implemented for the Set14 dataset using the ```Set14``` dataset class, but can be extended any of the other datasets with the right loaders. The prediction code is present in the ```predict.py``` file and contains the following required arguments -

* ```--data_root``` - the root directory for the dataset to run the prediction on
* ```--scale``` - the super-resolution scale of the output, which can be 2, 3, or 4
* ```--model_file``` - path to the .pt file to load the model from (has to correspond with the ```--scale``` parameter)
* ```--results_dir``` - path to the folder where the super-resolved images should be stored

An example command for testing is -

```
python predict.py --data_root Set14/ --scale 2 --model_file models/x2.pt --results_dir images/
```

### Summary of the architecture

A detailed description of the architecture can be found in the paper. The methods in the paper boil down to three aspects -

* Multiscale residual blocks - The network consists of a series of residual blocks that extract features from ```3x3``` and ```5x5``` convolution kernels simultaneously. This ensures that captures information in the images that may be scaled differently. These blocks have a residual element as well as feature concatenation between the two kernel outputs
* Fusing hierarchical features - The output of each residual block is combined in a bottleneck layer, which is ```1x1``` convlutional layer before the reconstruction module
* Pixel shuffle - Reconstruction is done using the [PixelShuffle](https://nico-curti.github.io/NumPyNet/NumPyNet/layers/pixelshuffle_layer.html) method

### Results

1. comic.png

    * __Original__

    ![](./images/comic.png)

    * __2X__

    ![](./images/comicx2.png)

    * __4X__

    ![](./images/comicx4.png)

2. lenna.png

    * __Original__

    ![](./images/lenna.png)

    * __2X__

    ![](./images/lennax2.png)

    * __4X__

    ![](./images/lennax4.png)
