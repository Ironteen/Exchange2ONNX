import os 
import tensorflow as tf
from absl import app, flags
from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model import tag_constants
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

FLAGS = flags.FLAGS
flags.DEFINE_string('mtype', "saved_model", 'save model type')
mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

sess = tf.InteractiveSession()

def main(argv):
    del argv
    mtype = FLAGS.mtype

    x = tf.placeholder(tf.float32, [None, 784], name="input")
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), 1))

    tf.identity(y, name="result")

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['result'])

    for i in range(20):
        batch_xs, batch_ys = mnist.train.next_batch(32)
        train_step.run({x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

    if "saved_model" in mtype:
        save_file = "./model/saved_model"
        if os.path.exists(save_file):
            import shutil
            shutil.rmtree(save_file)

        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder("./model/saved_model")
        signature = predict_signature_def(inputs={'input': x},
                                        outputs={'result': y})
        builder.add_meta_graph_and_variables(sess=sess,
                                            tags=[tag_constants.SERVING],
                                            signature_def_map={'predict': signature})
        builder.save()
    elif "ckpt" in mtype:
        ckpt_file = "./model/ckpt/"
        if not os.path.exists(ckpt_file):
            os.mkdir(ckpt_file)
        saver.save(sess,ckpt_file+"model.ckpt")
    elif "pb" in mtype:
        with tf.gfile.FastGFile('./model/model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())

if __name__=="__main__":
    app.run(main)