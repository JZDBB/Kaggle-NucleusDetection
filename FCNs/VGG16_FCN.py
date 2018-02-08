import tensorflow as tf

def weight_variable(shape):
  initial = tf.random_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(x, shape):
    initial = tf.constant(x, shape=shape)
    return tf.Variable(initial)

def conv2d(x, kernel, dim1, dim2):
    W = weight_variable([kernel, kernel, dim1, dim2])
    b = bias_variable(0., [dim2])
    y = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],
                          padding="SAME") + b
    output = tf.nn.relu(y)
    return output

def VGG16(x):
    with tf.name_scope("reshape"):
        x_image = tf.reshape(x, [-1, 224, 224, 3])

    with tf.name_scope("layers1"):
        with tf.name_scope("conv1"):
            conv1_1 = conv2d(x_image, 3, 3, 64)

        with tf.name_scope("conv2"):
            conv1_2 = conv2d(conv1_1, 3, 64, 64)

        with tf.name_scope("pool1"):
            y1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding="VALID")

    with tf.name_scope("layers2"):
        with tf.name_scope("conv1"):
            conv2_1 = conv2d(y1, 3, 64, 128)

        with tf.name_scope("conv2"):
            conv2_2 = conv2d(conv2_1, 3, 128, 128)

        with tf.name_scope("conv3"):
            conv2_3 = conv2d(conv2_2, 3, 128, 128)

        with tf.name_scope("pool2"):
            y2 = tf.nn.max_pool(conv2_3, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding="VALID")

    with tf.name_scope("layers3"):
        with tf.name_scope("conv1"):
            conv3_1 = conv2d(y2, 3, 128, 256)

        with tf.name_scope("conv2"):
            conv3_2 = conv2d(conv3_1, 3, 256, 256)

        with tf.name_scope("conv3"):
            conv3_3 = conv2d(conv3_2, 3, 256, 256)

        with tf.name_scope("pool3"):
            y3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding="VALID")

    with tf.name_scope("layers4"):
        with tf.name_scope("conv1"):
            conv4_1 = conv2d(y3, 3, 256, 512)

        with tf.name_scope("conv2"):
            conv4_2 = conv2d(conv4_1, 3, 512, 512)

        with tf.name_scope("conv3"):
            conv4_3 = conv2d(conv4_2, 3, 512, 512)

        with tf.name_scope("pool4"):
            y4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding="VALID")

    with tf.name_scope("layers5"):
        with tf.name_scope("conv1"):
            conv5_1 = conv2d(y4, 3, 512, 512)

        with tf.name_scope("conv2"):
            conv5_2 = conv2d(conv5_1, 3, 512, 512)

        with tf.name_scope("conv3"):
            conv5_3 = conv2d(conv5_2, 3, 128, 128)

        with tf.name_scope("pool5"):
            y5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding="VALID")
    with tf.name_scope("upsample1"):
