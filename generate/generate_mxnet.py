import mxnet as mx
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet
# Download pretrained resnet model - json and params from mxnet model zoo.
path='http://data.mxnet.io/models/imagenet/'
url1 = path+'resnet/18-layers/resnet-18-0000.params'
url2 = path+'resnet/18-layers/resnet-18-symbol.json'
url3 = path+'synset.txt'

[mx.test_utils.download(url1),mx.test_utils.download(url2),mx.test_utils.download(url3)]

sym = './model/resnet-18/resnet-18-symbol.json'
params = './model/resnet-18/resnet-18-0000.params'
# Standard Imagenet input - 3 channels, 224*224
input_shape = (1,3,224,224)
# Path of the output file
onnx_file = './model/mxnet_exported_resnet50.onnx'
converted_model_path = onnx_mxnet.export_model(sym, params, [input_shape], np.float32, onnx_file)
print(converted_model_path)