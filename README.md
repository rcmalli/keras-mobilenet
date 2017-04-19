# keras-MobileNet
Google MobileNet Implementation using Keras Functional Framework 2.0


### Project Summary

- This project is just the implementation of paper from scratch. I don't have the pretrained weights or GPU's to train :)
- Separable Convolution is already implemented in both keras and TF but, no BN support after Depthwise layers.
- Custom Depthwise Layer is just implemented by changing the source code of Separable Convolution from Keras sources [Keras: Separable Convolution](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L806)
- There is probably a typo in Table 1 at the last "Conv dw" layer stride should be 1 according to input sizes.

### TODO
- [x] Add Custom Depthwise Convolution
- [x] Add BN + RELU layers
- [x] Check layer shapes
- [ ] Test Custom Depthwise Convolution
- [ ] Benchmark Training and Feedforward with both CPU and GPU
- [ ] Compare with [SqueezeNet](https://github.com/rcmalli/keras-squeezenet)

### Library Versions

- Keras v2.0+
- Tensorflow 1.0+ (not supporting Theano for now)



### References

1) [Keras Framework](www.keras.io)

2) [Google MobileNet Paper](https://arxiv.org/pdf/1704.04861.pdf)


### Licence 

MIT License 

Note: If you find this project useful, please include reference link in your work.
