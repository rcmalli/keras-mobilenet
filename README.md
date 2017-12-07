
## Note: This project is not maintained anymore. Mobilenet implementation is already included in Keras Applications folder. [Mobilenet](https://github.com/fchollet/keras/blob/master/keras/applications/mobilenet.py)

# Keras MobileNet
Google MobileNet Implementation using Keras Framework 2.0

### Project Summary

- This project is just the implementation of paper from scratch. I don't have the pretrained weights or GPU's to train :)
- Separable Convolution is already implemented in both Keras and TF but, there is no BN support after Depthwise layers (Still investigating).
- Custom Depthwise Layer is just implemented by changing the source code of Separable Convolution from Keras. [Keras: Separable Convolution](https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L806)
- There is probably a typo in Table 1 at the last "Conv dw" layer stride should be 1 according to input sizes.
- Couldn't find any information about the usage of biases at layers (not used as default).

### TODO
- [x] Add Custom Depthwise Convolution
- [x] Add BN + RELU layers
- [x] Check layer shapes
- [ ] Test Custom Depthwise Convolution
- [ ] Benchmark training and feedforward pass with both CPU and GPU
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
