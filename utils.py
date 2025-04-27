import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def conv_layer_2d(x, filters, kernel_size, stride, trainable=True, name=None):
    return tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding='same',
        trainable=trainable,
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
        bias_initializer=tf.keras.initializers.GlorotUniform(),
        name=name)(x)

def deconv_layer_2d(x, filters, kernel_size, stride, output_shape=None, trainable=True, name=None):
    x = tf.pad(x, [[0,0], [3,3], [3,3], [0,0]], mode='reflect')
    x = tf.keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding='same',
        trainable=trainable,
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
        bias_initializer=tf.keras.initializers.GlorotUniform(),
        name=name)(x)
    return x[:, 3:-3, 3:-3, :]

def flatten_layer(x):
    input_shape = x.shape
    dim = input_shape[1] * input_shape[2] * input_shape[3]
    transposed = tf.transpose(x, (0, 3, 1, 2))
    return tf.reshape(transposed, [-1, dim])

def dense_layer(x, units, trainable=True, name=None):
    return tf.keras.layers.Dense(
        units=units,
        trainable=trainable,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        bias_initializer=tf.keras.initializers.Constant(0.0),
        name=name)(x)

def pixel_shuffle_layer(x, r, n_split):
    def PS(x, r):
        N, h, w = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        x = tf.reshape(x, (N, h, w, r, r))
        x = tf.transpose(x, (0, 1, 2, 4, 3))
        x = tf.split(x, h, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
        x = tf.split(x, w, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
        return tf.reshape(x, (N, h*r, w*r, 1))

    xc = tf.split(x, n_split, 3)
    return tf.concat([PS(x_, r) for x_ in xc], 3)

def plot_SR_data(idx, LR, SR, path):
    for i in range(LR.shape[0]):
        vmin0, vmax0 = np.min(SR[i,:,:,0]), np.max(SR[i,:,:,0])
        vmin1, vmax1 = np.min(SR[i,:,:,1]), np.max(SR[i,:,:,1])

        plt.figure(figsize=(12, 12))
        
        plt.subplot(221)
        plt.imshow(LR[i, :, :, 0], vmin=vmin0, vmax=vmax0, cmap='viridis', origin='lower')
        plt.title('LR 0 Input', fontsize=9)
        plt.colorbar()
        plt.xticks([], [])
        plt.yticks([], [])
        
        plt.subplot(223)
        plt.imshow(LR[i, :, :, 1], vmin=vmin1, vmax=vmax1, cmap='viridis', origin='lower')
        plt.title('LR 1 Input', fontsize=9)
        plt.colorbar()
        plt.xticks([], [])
        plt.yticks([], [])
        
        plt.subplot(222)
        plt.imshow(SR[i, :, :, 0], vmin=vmin0, vmax=vmax0, cmap='viridis', origin='lower')
        plt.title('SR 0 Output', fontsize=9)
        plt.colorbar()
        plt.xticks([], [])
        plt.yticks([], [])
        
        plt.subplot(224)
        plt.imshow(SR[i, :, :, 1], vmin=vmin1, vmax=vmax1, cmap='viridis', origin='lower')
        plt.title('SR 1 Output', fontsize=9)
        plt.colorbar()
        plt.xticks([], [])
        plt.yticks([], [])

        plt.savefig(path+'/img{0:05d}.png'.format(idx[i]), dpi=200, bbox_inches='tight')
        plt.close()

def downscale_image(x, K):
    if x.ndim == 3:
        x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))

    weight = tf.constant(1.0/K**2, shape=[K, K, x.shape[3], x.shape[3]], dtype=tf.float64)
    return tf.nn.conv2d(x, filters=weight, strides=[1, K, K, 1], padding='SAME')

def generate_TFRecords(filename, data, mode='test', K=None):
    if mode == 'train':
        assert K is not None, 'In training mode, downscaling factor K must be specified'
        data_LR = downscale_image(data, K)

    with tf.io.TFRecordWriter(filename) as writer:
        for j in range(data.shape[0]):
            if mode == 'train':
                h_HR, w_HR, c = data[j, ...].shape
                h_LR, w_LR, c = data_LR[j, ...].shape
                features = tf.train.Features(feature={
                                     'index': tf.train.Feature(int64_list=tf.train.Int64List(value=[j])),
                                   'data_LR': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_LR[j, ...].tostring()])),
                                      'h_LR': tf.train.Feature(int64_list=tf.train.Int64List(value=[h_LR])),
                                      'w_LR': tf.train.Feature(int64_list=tf.train.Int64List(value=[w_LR])),
                                   'data_HR': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data[j, ...].tostring()])),
                                      'h_HR': tf.train.Feature(int64_list=tf.train.Int64List(value=[h_HR])),
                                      'w_HR': tf.train.Feature(int64_list=tf.train.Int64List(value=[w_HR])),
                                         'c': tf.train.Feature(int64_list=tf.train.Int64List(value=[c]))})
            elif mode == 'test':
                h_LR, w_LR, c = data[j, ...].shape
                features = tf.train.Features(feature={
                                     'index': tf.train.Feature(int64_list=tf.train.Int64List(value=[j])),
                                   'data_LR': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data[j, ...].tostring()])),
                                      'h_LR': tf.train.Feature(int64_list=tf.train.Int64List(value=[h_LR])),
                                      'w_LR': tf.train.Feature(int64_list=tf.train.Int64List(value=[w_LR])),
                                         'c': tf.train.Feature(int64_list=tf.train.Int64List(value=[c]))})

            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())