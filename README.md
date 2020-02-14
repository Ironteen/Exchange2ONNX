# Model Exchange To ONNX Format

[Open Neural Network Exchange (ONNX)](https://onnx.ai/) is an open ecosystem that empowers AI developers to choose the right tools as their project evolves. ONNX provides an open source format for AI models, both deep learning and traditional ML. It defines an extensible computation graph model, as well as definitions of built-in operators and standard data types. Enabling interoperability between different frameworks and streamlining the path from research to production helps increase the speed of innovation in the AI community.

This program is a secondary development project based on onnx and the work of my research group.

#### Support Model

| Framework / Tool                                     | Model Format            | Argparse    | Notes            |
| ---------------------------------------------------- | ----------------------- | ----------- | ---------------- |
| [TensorFlow](https://www.tensorflow.org/)            | .pb                     | 1-4, 6, 8   |                  |
| [TensorFlow](https://www.tensorflow.org/)            | .ckpt.meta              | 1-4, 6, 8   |                  |
| [TensorFlow](https://www.tensorflow.org/)            | saved_model             | 1-4, 6-8    |                  |
| [Keras](https://github.com/keras-team/keras)         | .h5                     | 1-2, 6, 8   |                  |
| [MXNet (Apache)](http://mxnet.incubator.apache.org/) | .params & .json         | 1-2, 5-6, 8 |                  |
| [Caffe](https://github.com/BVLC/caffe)               | .prototxt & cafffemodel | 1-2, 6, 8   | for Linux/macos  |
| [PyTorch](http://pytorch.org/)                       | .pth                    | 1-2, 5-6, 8 | require NN class |

##### Argparse:

| Index | Name          | Type        | Default               | Note                           |
| ----- | ------------- | ----------- | --------------------- | ------------------------------ |
| 1     | model         | STRING      | None                  | init FP32 saved model          |
| 2     | output        | STRING      | "./output/model.onnx" | output model file              |
| 3     | input_name    | STRING_LIST | ["input:0"]           | model input_names              |
| 4     | output_name   | STRING_LIST | ["result:0"]          | model input_names              |
| 5     | input_size    | LIST        | [1,3,224,224]         | input data size                |
| 6     | validate      | BOOL        | True                  | validate onnx model            |
| 7     | signature_def | STRING      | None                  | signature_def from saved model |
| 8     | opset         | INT         | 7                     | opset version to use for onnx  |

#### Requirements

We need to install some libraries required

```
pip install -r requirements.txt
```

#### Generate model for test

- TensorFlow

  ```
  python ./generate/generate_tfmodel.py --mtype=ckpt   # ckpt or pb or saved_model
  ```

- Keras

  ```
  python ./generate/generate_keras.py
  ```

- Pytorch

  ```
  python ./generate/generate_pth.py
  ```

- MXNet

  ```
  python ./generate/generate_mxnet.py
  ```

#### Convert

- **TensorFlow**

  .pb and ckpt

  ```
  python convert2onnx.py --model="./model/ckpt/model.ckpt.meta" --output="./output/model.onnx" --input_name=["input:0"] --output_name=["result:0"] --validate=True --opset=7
  ```

  saved_model

  ```
  python convert2onnx.py --model="./model/saved_model" --output="./output/model.onnx" --input_name=["input:0"] --output_name=["result:0"] --signature_def=None --validate=True --opset=7
  ```

- **Keras**

  ```
  python convert2onnx.py --model="./model/mnistmodel.h5" --output="./output/model.onnx" --validate=True --opset=7
  ```

- **MXNet**

  ```
  python convert2onnx.py --model="./model/mxnet-model" --output="./output/model.onnx" --input_size=[1,3,3,224] --validate=True --opset=7
  ```

- **Caffe**

  ```
  python convert2onnx.py --model="./model/caffe-model" --output="./output/model.onnx" --validate=True --opset=7
  ```

- **Pytorch**

  ```
  python convert2onnx.py --model="./model/model.pth" --output="./output/model.onnx" --input_size=[1,3,3,224] --validate=True --opset=7
  ```

#### Reference

- [ONNX Tutorials](https://github.com/onnx/tutorials)