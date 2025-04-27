import tensorflow as tf
from utils import *

class Generator(tf.keras.Model):
    def __init__(self, r, C):
        super(Generator, self).__init__()
        self.r = r
        self.C = C
        self.k = 3
        self.stride = 1
        
        # Initial layers
        self.deconv1 = tf.keras.layers.Conv2DTranspose(
            64, self.k, strides=self.stride, padding='same', 
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            activation='relu')
        
        # Residual blocks
        self.res_blocks = []
        for i in range(16):
            self.res_blocks.append([
                tf.keras.layers.Conv2DTranspose(
                    64, self.k, strides=self.stride, padding='same',
                    kernel_initializer=tf.keras.initializers.GlorotUniform(),
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    64, self.k, strides=self.stride, padding='same',
                    kernel_initializer=tf.keras.initializers.GlorotUniform())
            ])
        
        # Final layers
        self.deconv2 = tf.keras.layers.Conv2DTranspose(
            64, self.k, strides=self.stride, padding='same',
            kernel_initializer=tf.keras.initializers.GlorotUniform())
        
        # Super resolution scaling
        self.scaling_layers = []
        for r_i in r:
            self.scaling_layers.append(tf.keras.layers.Conv2DTranspose(
                (r_i**2)*64, self.k, strides=self.stride, padding='same',
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                activation='relu'))
        
        self.deconv_out = tf.keras.layers.Conv2DTranspose(
            C, self.k, strides=self.stride, padding='same',
            kernel_initializer=tf.keras.initializers.GlorotUniform())

    @tf.autograph.experimental.do_not_convert
    def call(self, x, training=False):
        # Initial deconv layer
        x = tf.pad(x, [[0,0], [3,3], [3,3], [0,0]], mode='reflect')
        x = self.deconv1(x)
        x = x[:, 3:-3, 3:-3, :]
        skip_connection = x
        
        # Residual blocks
        for block_a, block_b in self.res_blocks:
            B_skip_connection = x
            
            x = tf.pad(x, [[0,0], [3,3], [3,3], [0,0]], mode='reflect')
            x = block_a(x)
            x = x[:, 3:-3, 3:-3, :]
            
            x = tf.pad(x, [[0,0], [3,3], [3,3], [0,0]], mode='reflect')
            x = block_b(x)
            x = x[:, 3:-3, 3:-3, :]
            
            x = tf.add(x, B_skip_connection)
        
        # Final deconv layer
        x = tf.pad(x, [[0,0], [3,3], [3,3], [0,0]], mode='reflect')
        x = self.deconv2(x)
        x = x[:, 3:-3, 3:-3, :]
        x = tf.add(x, skip_connection)
        
        # Super resolution scaling
        r_prod = 1
        for i, (r_i, layer) in enumerate(zip(self.r, self.scaling_layers)):
            x = tf.pad(x, [[0,0], [3,3], [3,3], [0,0]], mode='reflect')
            x = layer(x)
            x = x[:, 3:-3, 3:-3, :]
            x = tf.nn.depth_to_space(x, r_i)
            r_prod *= r_i
        
        # Final output layer
        x = tf.pad(x, [[0,0], [3,3], [3,3], [0,0]], mode='reflect')
        x = self.deconv_out(x)
        x = x[:, 3:-3, 3:-3, :]
        
        return x

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(
            32, 3, strides=1, padding='same',
            kernel_initializer=tf.keras.initializers.GlorotUniform())
        
        self.conv2 = tf.keras.layers.Conv2D(
            32, 3, strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.GlorotUniform())
        
        self.conv3 = tf.keras.layers.Conv2D(
            64, 3, strides=1, padding='same',
            kernel_initializer=tf.keras.initializers.GlorotUniform())
        
        self.conv4 = tf.keras.layers.Conv2D(
            64, 3, strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.GlorotUniform())
        
        self.conv5 = tf.keras.layers.Conv2D(
            128, 3, strides=1, padding='same',
            kernel_initializer=tf.keras.initializers.GlorotUniform())
        
        self.conv6 = tf.keras.layers.Conv2D(
            128, 3, strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.GlorotUniform())
        
        self.conv7 = tf.keras.layers.Conv2D(
            256, 3, strides=1, padding='same',
            kernel_initializer=tf.keras.initializers.GlorotUniform())
        
        self.conv8 = tf.keras.layers.Conv2D(
            256, 3, strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.GlorotUniform())
        
        self.flatten = tf.keras.layers.Flatten()
        
        self.fc1 = tf.keras.layers.Dense(
            1024, kernel_initializer=tf.keras.initializers.GlorotUniform())
        
        self.fc2 = tf.keras.layers.Dense(
            1, kernel_initializer=tf.keras.initializers.GlorotUniform())

    def call(self, x, training=False):
        x = self.conv1(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        x = self.conv2(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        x = self.conv3(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        x = self.conv4(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        x = self.conv5(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        x = self.conv6(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        x = self.conv7(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        x = self.conv8(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        
        x = self.fc2(x)
        
        return x

class SR_NETWORK:
    def __init__(self, r, C, status='pretraining', alpha_advers=0.001):
        status = status.lower()
        if status not in ['pretraining', 'training', 'testing']:
            print('Error in network status.')
            exit()

        self.r = r
        self.C = C
        self.status = status
        self.alpha_advers = alpha_advers
        
        self.generator = Generator(r, C)
        
        if status in ['pretraining', 'training']:
            self.discriminator = Discriminator()
        
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        if status == 'training':
            self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    def compute_losses(self, x_HR, x_SR, d_HR=None, d_SR=None):
        content_loss = tf.reduce_mean((x_HR - x_SR)**2, axis=[1, 2, 3])

        if self.status == 'training':
            g_advers_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_SR, labels=tf.ones_like(d_SR))

            d_advers_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=tf.concat([d_HR, d_SR], axis=0),
                labels=tf.concat([tf.ones_like(d_HR), tf.zeros_like(d_SR)], axis=0))

            advers_perf = [
                tf.reduce_mean(tf.cast(tf.sigmoid(d_HR) > 0.5, tf.float32)),  # % true positive
                tf.reduce_mean(tf.cast(tf.sigmoid(d_SR) < 0.5, tf.float32)),  # % true negative
                tf.reduce_mean(tf.cast(tf.sigmoid(d_SR) > 0.5, tf.float32)),  # % false positive
                tf.reduce_mean(tf.cast(tf.sigmoid(d_HR) < 0.5, tf.float32))   # % false negative
            ]

            g_loss = tf.reduce_mean(content_loss) + self.alpha_advers * tf.reduce_mean(g_advers_loss)
            d_loss = tf.reduce_mean(d_advers_loss)

            return g_loss, d_loss, advers_perf, content_loss, g_advers_loss
        else:
            return tf.reduce_mean(content_loss)

    @tf.function
    def train_step(self, batch_LR, batch_HR):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            batch_SR = self.generator(batch_LR, training=True)
            
            if self.status == 'pretraining':
                g_loss = self.compute_losses(batch_HR, batch_SR)
                d_loss, advers_perf = None, None
            else:
                disc_real = self.discriminator(batch_HR, training=True)
                disc_fake = self.discriminator(batch_SR, training=True)
                
                g_loss, d_loss, advers_perf, content_loss, g_advers_loss = self.compute_losses(
                    batch_HR, batch_SR, disc_real, disc_fake)

        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        if self.status == 'training':
            gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return g_loss, d_loss, advers_perf

    def generate(self, x_LR, training=False):
        return self.generator(x_LR, training=training)