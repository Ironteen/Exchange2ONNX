#!/usr/bin/env python
# -*- coding:utf-8 -*-
# title            : convert2onnx.py
# description      : convert model to onnx format
# author           : Zhijun Tu
# email            : tzj19970116@163.com
# date             : 2020/02/10
# version          : 1.0
# notes            : suport TensorFlow, Pytorch, caffee and mxnet
# python version   : 3.5 or later version
############################################################### 
import os
import onnx
from onnx import checker
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('model', None, 'init FP32 saved model')
flags.DEFINE_string('output', "./output/model.onnx", 'output model file')
flags.DEFINE_list('input_name', ["input:0"], 'model input_names')
flags.DEFINE_list('output_name', ["result:0"], 'model output_names')
flags.DEFINE_list('input_size', [1,3,224,224], 'input data size')
flags.DEFINE_bool('validate', True, 'validate onnx model')
flags.DEFINE_string('signature_def', None, 'signature_def from saved model to use')
flags.DEFINE_integer('opset', 7, 'opset version to use for onnx domain')

save_path = "./output"
model_path = "./model"
if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(model_path):
    os.mkdir(model_path)

# convert caffe to onnx
def convert_caffee2onnx(proto,caffemodel,output):
    '''
    It is not supported in Wondows,for the library 
    -- "coremltools" are only for linux and macos
    '''
    import coremltools
    import onnxmltools
    output_coreml_model = './model/model.mlmodel'

    output_onnx_model = 'model.onnx'
    coreml_model = coremltools.converters.caffe.convert((caffemodel, proto))
    coreml_model.save(output)
    coreml_model = coremltools.utils.load_spec(output)
    onnx_model = onnxmltools.convert_coreml(coreml_model)
    onnxmltools.utils.save_model(onnx_model, output)

# convert tensorflow to onnx
def convert_tf2onnx(model,output,inputs,outputs,signature_def=None, opset=7):
    import tensorflow as tf
    from tf2onnx.tfonnx import process_tf_graph, tf_optimize
    from tf2onnx import constants, loader, logging, utils, optimizer
    logger = logging.getLogger(constants.TF2ONNX_PACKAGE_NAME)

    if "pb" in model:
        graph_def, inputs, outputs = loader.from_graphdef(model, inputs, outputs)
    elif "meta" in model:
        graph_def, inputs, outputs = loader.from_checkpoint(model, inputs, outputs)
    elif "saved_model" in model:
        graph_def, inputs, outputs = loader.from_saved_model(model, inputs, outputs, signature_def)

    graph_def = tf_optimize(inputs, outputs, graph_def, None)
    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graph_def, name='')
    with tf.Session(graph=tf_graph):
        g = process_tf_graph(tf_graph,opset=opset,input_names=inputs,output_names=outputs)

    onnx_graph = optimizer.optimize_graph(g)
    model_proto = onnx_graph.make_model("converted from {}".format(model))
    # write onnx graph
    logger.info("")
    logger.info("Successfully converted TensorFlow model %s to ONNX", model)
    utils.save_protobuf(output, model_proto)
    logger.info("ONNX model is saved at %s", output)

# convert keras to onnx
def convert_keras2onnx(model,output):
    import onnxmltools
    from keras.models import load_model

    keras_model = load_model(model)
    onnx_model = onnxmltools.convert_keras(keras_model)
    onnxmltools.utils.save_model(onnx_model, output)

# convert pytorch to onnx
def convert_pth2onnx(model_path,output,input_size=[1,3,224,224]):
    import torch
    import torch.onnx
    import torchvision
    from torch.autograd import Variable

    dummy_model_input = Variable(torch.randn(input_size))
    print("pth:",model_path)
    model = torch.load(model_path)
    torch.onnx.export(model, dummy_model_input, output)

# convert mxnet to onnx
def convert_mxnet2onnx(params,json,output,input_shape = [1,3,224,224]):
    import mxnet as mx
    import numpy as np
    from mxnet.contrib import onnx as onnx_mxnet

    converted_model_path = onnx_mxnet.export_model(json, params, [input_shape], np.float32, output)

# validate the onnx model
def Check_validity(onnx_model):
    model_proto = onnx.load(onnx_model)
    checker.check_graph(model_proto.graph)

def main(argv):
    del argv
    model         = FLAGS.model
    output        = FLAGS.output
    input_name    = FLAGS.input_name
    output_name   = FLAGS.output_name
    input_size    = FLAGS.input_size
    isvaladate    = FLAGS.validate
    signature_def = FLAGS.signature_def
    opset         = FLAGS.opset
    data          = "./data/image.npz."

    if not os.path.exists(model):
        assert("%s not found"%model)
    else:
        print("model path: ",model)
    # saved_model, mxnet_model
    filedict = dict()
    if os.path.isdir(model):
        files = os.listdir(model)
        for file in files:
            if "variables" in file:
                convert_tf2onnx(model,output,input_name,output_name,signature_def,opset)
                break
            elif "params" in file:
                filedict["params"] = os.path.join(model,file)
            elif "json" in file:
                filedict["json"] = os.path.join(model,file)
            elif "prototxt" in file:
                filedict["prototxt"] = os.path.join(model,file)
            elif "caffemodel" in file:
                filedict["caffemodel"] = os.path.join(model,file)
        if "params" in filedict.keys() and "json" in filedict.keys():
            convert_mxnet2onnx(filedict['params'],filedict['json'],output,input_size)
        elif "prototxt" in filedict.keys() and "caffemodel" in filedict.keys():
            convert_caffee2onnx(filedict['prototxt'],filedict['caffemodel'],output)
    else:
        # .pb or chechpoint/meta file
        if ".pb" in model or "meta" in model:   
            convert_tf2onnx(model,output,input_name,output_name,signature_def,opset)
        elif ".h5" in model:
            convert_keras2onnx(model,output)
        elif ".pth" in model:
            convert_pth2onnx(model,output,input_size)

    if isvaladate:
        Check_validity(output)

if __name__=="__main__":
    app.run(main)

