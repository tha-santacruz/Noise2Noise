# Noise2Noise
 Super resolution UNET model trained with low resolution pairs only
 The two miniprojects were made in the context of the EE-559 Deep Learning course of EPFL during spring 2022.
 The model architectures are based on [[1]](#1)
## Miniproject 1
 Contains an implementation of the reference model in pytorch, as well as model weights trained on pairs of noisy subimages for imagenet
 
## Miniproject 2
 In this project, a similar architecture is proposed without using standard pytorch modules or autograd (for educational purposes). The modules that have been recreated in this project are : Mean Squared Error loss, ReLU, Sigmoid, Nearest Neighbor Upsampling, Conv2d, Upsampling (or transpose convolution), and Sequential (allowing to stack modules). The implementation of these modules include forward, and backward pass. In adition to that, the Stochastic Gradient Descend optimizer has also been coded, with the step and zero_grad methods.
 
## Prediction Example
![plot](./Miniproject_1/others/results_examples/result_1/results7.png)

 
 ## References

<a id="1">[1]</a>  Jaakko Lehtinen, Jacob Munkberg, Jon Hasselgren, Samuli Laine, Tero Karras, Miika Aittala, Timo Aila. Noise2Noise: Learning Image Restoration without Clean Data, 2018, https://doi.org/10.48550/arxiv.1803.04189
