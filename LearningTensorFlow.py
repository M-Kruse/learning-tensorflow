
import os
#Set the log level to get rid of the noisy tensorflow console messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" #Uncomment this to force tf to run on the CPU when a GPU is available
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import subprocess


class trainerCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>=0.90):
          print("\nReached 90% accuracy so cancelling training!")
          self.model.stop_training = True

class LearningTensorFlow(object):
    """docstring for LearningTensorFlow"""
    def __init__(self):
        super(LearningTensorFlow, self).__init__()

    def misc_info(self):
        print("Eagerly mode: {0}".format(tf.executing_eagerly()))
        print("GPU Detected: {0}")
        print("CUDA Support: {0}".format(tf.test.is_built_with_cuda()))

    def get_device_list(self):
        from tensorflow.python.client import device_lib
        return device_lib.list_local_devices()

    def test_device(self, compute_device):
        with tf.compat.v1.Session() as sess:
            with tf.device(compute_device):
                foo = tf.constant([8.0, 14.0, 15.0], shape=[1, 3], name='foo')
                bar = tf.constant([2.0, 1.0, 26.0], shape=[3, 1], name='bar')
                baz = tf.matmul(foo, bar)
                print(" Running Test...")
                #print(p.numpy())
                result = sess.run(baz)
                if result[0][0] == 420.0:
                    return True
                else:
                    return False

    def example_model_fitting():
        model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

        model.compile(optimizer='sgd', loss='mean_squared_error')

        xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
        ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

        model.fit(xs, ys, epochs=500) #[[18.975805]]
        #model.fit(xs, ys, epochs=50000) #[[18.999987]]

        print(model.predict([10.0]))
        #tf.compat.v1.Session(config=self.tf_config) as sess:


    def example_nn_classifier(self):
        callbacks = trainerCallback()
    
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
        train_images = train_images/255.0
        test_images = test_images/255.0

        model = keras.Sequential([
                keras.layers.Flatten(input_shape=(28,28)),
                keras.layers.Dense(128, activation=tf.nn.relu),
                keras.layers.Dense(10, activation=tf.nn.softmax)
            ])

        model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

        model.fit(train_images, train_labels, epochs=50, callbacks=[callbacks])

        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print("Training Metrics:")
        print(" Loss: {}".format(test_loss))
        print(" Accuracy: {}".format(test_acc))
        
        test_images = test_images / 255.0
        predictions = model.predict(test_images)
        print("Prediction Test: ")
        print(" {0}".format(predictions[34]))
        print(" {0}".format(test_labels[34]))

    def example_cnn_classifier(self):
        callbacks = trainerCallback()
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        
        train_images = train_images.reshape(60000, 28, 28, 1) #cnn
        train_images = train_images/255.0
        test_images = test_images.reshape(10000, 28, 28, 1) #cnn
        test_images = test_images/255.0

        model = keras.Sequential([
                keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)), #cnn
                keras.layers.MaxPooling2D(2,2), #cnn
                keras.layers.Conv2D(64, (3, 3), activation='relu'), #cnn
                keras.layers.MaxPooling2D(2,2), #cnn
                keras.layers.Flatten(input_shape=(28,28)),
                keras.layers.Dense(128, activation=tf.nn.relu),
                keras.layers.Dense(10, activation=tf.nn.softmax)
            ])

        model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

        model.fit(train_images, train_labels, epochs=50, callbacks=[callbacks])

        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print("Loss: {}".format(test_loss))
        print("Accuracy: {}".format(test_acc))
        
        test_images = test_images / 255.0
        predictions = model.predict(test_images)

        print(predictions[34])
        print(test_labels[34])

    def show_numpy_backend(self):
        print(np.show_config())

    def naive_vector_dot(self, x, y):
        assert len(x.shape) == 1
        assert len(y.shape) == 1
        assert x.shape[0] == y.shape[0]
        z = 0.
        for i in range(x.shape[0]):
            z += x[i] * y[i]
        return z

    def tensor_examples(self):
        s = np.array(42)
        print("{0} Dimension Tensor (Scalar):\n {1}".format(s.ndim, s))
        v = np.array([1, 2, 3, 4])
        print("{0} Dimension Tensor (Vector):\n {1}".format(v.ndim, v))
        #print(v.ndim)
        m = np.array([[1, 4, 7, 10, 13],
                      [2, 5, 8, 11, 14],
                      [3, 6, 9, 12, 15]])
        print("{0} Dimension Tensor (Matrix):\n {1}".format(m.ndim, m))
        t = np.array([[[1, 4, 7, 10, 13],
                      [2, 5, 8, 11, 14],
                      [3, 6, 9, 12, 15]],
                     [[1, 4, 7, 10, 13],
                      [2, 5, 8, 11, 14],
                      [3, 6, 9, 12, 15]],
                     [[1, 4, 7, 10, 13],
                      [2, 5, 8, 11, 14],
                      [3, 6, 9, 12, 15]]])
        print("{0} Dimension Tensor (3D Tensor):\n {1}".format(t.ndim, t))

    def example_vector_dot(self, vec1, vec2):
        assert len(vec1.shape) == 1
        assert len(vec2.shape) == 1
        assert vec1.shape[0] == vec2.shape[0]
        dot = 0.
        print("Vector 1: {0}".format(vec1))
        print("Vector 2: {0}".format(vec2))
        print()
        for i in range(vec1.shape[0]):
            print("Vector 1 Slice: {0}".format(vec1[i]))
            print("Vector 2 Slice: {0}".format(vec2[i]))
            print("Running Dot Before Op: {0}".format(dot))
            dot += vec1[i] * vec2[i]
            print("Running Dot After Op: {0}".format(dot))
        print("Final Dot Product (Scalar): {0}".format(dot))
        return dot

    
    def example_matrix_vector_dot(self, matrix, vector):
        assert len(matrix.shape) == 2
        assert len(vector.shape) == 1
        assert matrix.shape[1] == vector.shape[0]
        dot = np.zeros(matrix.shape[0])
        print("Matrix:\n{0}".format(matrix))
        print("Vector: {0}".format(vector))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                print("Current Matrix Slice: {0}".format(matrix[i, j]))
                print("Current Vector Slice: {0}".format(vector[j]))
                print("Running Dot Before Op: {0}".format(dot))
                dot[i] += matrix[i, j] * vector[j]
                print("Running Dot After Op: {0}".format(dot))
        print("Final Matrix Dot Product: {0}".format(dot))

    def example_matrix_dot(self, matrix1, matrix2):
        assert len(matrix1.shape) == 2
        assert len(matrix2.shape) == 2
        assert matrix1.shape[1] == matrix2.shape[0]
        dot = np.zeros((matrix1.shape[0], matrix2.shape[1]))
        for i in range(matrix1.shape[0]):
            for j in range(matrix2.shape[1]):
                    row_matrix1 = matrix1[i, :]
                    column_matrix2 = matrix2[:, j]
                    print("Running Dot Before Op: {0}".format(dot))
                    dot[i, j] = self.example_vector_dot(row_matrix1, column_matrix2)
                    print("Running Dot After Op: {0}".format(dot))
        return dot


if __name__ == '__main__':

    ltf = LearningTensorFlow()

    print("TensorFlow Device Testing...\n")

    devs = ltf.get_device_list()
    for d in devs:
        print(" Device Name: {}".format(d.name))
        print(" Device Type: {}".format(d.device_type))
        print(" Passed Test: {}".format(ltf.test_device(d.name)))
        print(" Memory Size: {}\n".format(d.memory_limit))


    print("Tensor object testing...\n")

    print(" Running equation: (2^2) + (2^2)")
    my_tensor = tf.square(2) + tf.square(2)
    print(" Tensor result: {0}".format(my_tensor))
    print(" Tensor obj info: {0}".format(type(my_tensor)))
    print(" Tensor dtype: {0}".format(my_tensor.dtype))
    print(" Device containing tensor: {0}".format(my_tensor.device))


    #ltf.tensor_examples()

    #Example vector dot product
    # v1 = np.array([1, 2, 3, 4])
    # v2 = np.array([1, 2, 3, 4])
    # ltf.example_vector_dot(v1, v2)

    #Example vector matrix dot product
    v3 = np.array([1, 2, 3])
    m1 = np.array([
                    [1, 4, 7],
                    [2, 5, 8],
                    [3, 6, 9]
                ])

    #ltf.example_matrix_vector_dot(m1, v3)

    m2 = np.array([
                        [1, 4, 7],
                        [2, 5, 8],
                        [3, 6, 9]
                    ])


    print(ltf.example_matrix_dot(m1, m2))



    # fashion_mnist = keras.datasets.fashion_mnist
    # (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


    # # scale the values to 0.0 to 1.0
    # train_images = train_images / 255.0
    # test_images = test_images / 255.0

    # # reshape for feeding into the model
    # train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    # test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    #                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
    # print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))

    # model = keras.Sequential([
    #   keras.layers.Conv2D(input_shape=(28,28,1), filters=8, kernel_size=3, 
    #                       strides=2, activation='relu', name='Conv1'),
    #   keras.layers.Flatten(),
    #   keras.layers.Dense(10, activation=tf.nn.softmax, name='Softmax')
    # ])
    # model.summary()

    # testing = False
    # epochs = 5

    # model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), 
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])
    # model.fit(train_images, train_labels, epochs=epochs)

    # test_loss, test_acc = model.evaluate(test_images, test_labels)
    # print('\nTest accuracy: {}'.format(test_acc))


    # model_zoo = "/home/devel/Code/my_model_zoo/tensorflow/"
    # model_name = "serve_test"
    # model_dir = model_zoo + model_name
    # version = 1
    # export_path = os.path.join(model_dir, str(version))
    # print('export_path = {}\n'.format(export_path))
    # if os.path.isdir(export_path):
    #   print('\nAlready saved a model, cleaning up\n')

    # tf.compat.v1.saved_model.simple_save(
    #     tf.compat.v1.keras.backend.get_session(),
    #     export_path,
    #     inputs={'input_image': model.input},
    #     outputs={t.name:t for t in model.outputs})

    # print('\nSaved model:')


# devel@devel:~/Code/learning-tensorflow$ ~/.local/bin/saved_model_cli show --dir ~/Code/my_model_zoo/tensorflow/serve_test/1/ --all

# MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

# signature_def['serving_default']:
#   The given SavedModel SignatureDef contains the following input(s):
#     inputs['input_image'] tensor_info:
#         dtype: DT_FLOAT
#         shape: (-1, 28, 28, 1)
#         name: Conv1_input:0
#   The given SavedModel SignatureDef contains the following output(s):
#     outputs['Softmax/Softmax:0'] tensor_info:
#         dtype: DT_FLOAT
#         shape: (-1, 10)
#         name: Softmax/Softmax:0
#   Method name is: tensorflow/serving/predict
