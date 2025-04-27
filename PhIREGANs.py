import os
import numpy as np
import tensorflow as tf
from time import strftime, time
from utils import plot_SR_data
from sr_network import SR_NETWORK

class PhIREGANs:
    DEFAULT_N_EPOCHS = 10
    DEFAULT_LEARNING_RATE = 1e-4
    DEFAULT_EPOCH_SHIFT = 0
    DEFAULT_SAVE_EVERY = 10
    DEFAULT_PRINT_EVERY = 2

    def __init__(self, data_type, N_epochs=None, learning_rate=None, epoch_shift=None, save_every=None, print_every=None, mu_sig=None):
        self.N_epochs = N_epochs if N_epochs is not None else self.DEFAULT_N_EPOCHS
        self.learning_rate = learning_rate if learning_rate is not None else self.DEFAULT_LEARNING_RATE
        self.epoch_shift = epoch_shift if epoch_shift is not None else self.DEFAULT_EPOCH_SHIFT
        self.save_every = save_every if save_every is not None else self.DEFAULT_SAVE_EVERY
        self.print_every = print_every if print_every is not None else self.DEFAULT_PRINT_EVERY

        self.data_type = data_type
        self.mu_sig = mu_sig
        self.LR_data_shape = None

        self.run_id = '-'.join([self.data_type, strftime('%Y%m%d-%H%M%S')])
        self.model_name = '/'.join(['models', self.run_id])
        self.data_out_path = '/'.join(['data_out', self.run_id])

    # ... (keep all the setter methods unchanged)

    def pretrain(self, r, data_path, model_path=None, batch_size=100):
        if self.mu_sig is None:
            self.set_mu_sig(data_path, batch_size)
        
        self.set_LR_data_shape(data_path)
        h, w, C = self.LR_data_shape

        print('Initializing network ...', end=' ')
        model = SR_NETWORK(r=r, C=C, status='pretraining')
        
        # Create checkpoint manager
        checkpoint = tf.train.Checkpoint(generator=model.generator)
        manager = tf.train.CheckpointManager(checkpoint, self.model_name, max_to_keep=5)
        
        if model_path is not None:
            print('Loading previously trained network...', end=' ')
            checkpoint.restore(model_path)
            print('Done.')

        print('Building data pipeline ...', end=' ')
        dataset = tf.data.TFRecordDataset(data_path)
        dataset = dataset.map(lambda xx: self._parse_train_(xx, self.mu_sig))
        dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        print('Done.')

        print('Training network ...')
        for epoch in range(self.epoch_shift+1, self.epoch_shift+self.N_epochs+1):
            print('Epoch: %d' % epoch)
            start_time = time()
            epoch_loss, N = 0, 0

            for batch in dataset:
                batch_idx, batch_LR, batch_HR = batch
                N_batch = batch_LR.shape[0]
                
                g_loss, _, _ = model.train_step(batch_LR, batch_HR)
                
                epoch_loss += g_loss.numpy() * N_batch
                N += N_batch

            epoch_loss = epoch_loss / N
            print('Epoch generator training loss=%.5f' % epoch_loss)
            print('Epoch took %.2f seconds\n' % (time() - start_time), flush=True)

            if (epoch % self.save_every) == 0:
                saved_model = manager.save()
                print(f"Saved model at {saved_model}")

        saved_model = manager.save()
        print('Done.')
        return saved_model

    def train(self, r, data_path, model_path, batch_size=100, alpha_advers=0.001):
        assert model_path is not None, 'Must provide path for pretrained model'
        
        if self.mu_sig is None:
            self.set_mu_sig(data_path, batch_size)
        
        self.set_LR_data_shape(data_path)
        h, w, C = self.LR_data_shape

        print('Initializing network ...', end=' ')
        model = SR_NETWORK(r=r, C=C, status='training', alpha_advers=alpha_advers)
        
        # Create checkpoint managers
        g_checkpoint = tf.train.Checkpoint(generator=model.generator)
        g_manager = tf.train.CheckpointManager(g_checkpoint, os.path.join(self.model_name, 'gan'), max_to_keep=5)
        
        gd_checkpoint = tf.train.Checkpoint(generator=model.generator, discriminator=model.discriminator)
        gd_manager = tf.train.CheckpointManager(gd_checkpoint, os.path.join(self.model_name, 'gan-all'), max_to_keep=5)
        
        print('Loading previously trained network...', end=' ')
        if 'gan-all' in model_path:
            gd_checkpoint.restore(model_path)
        else:
            g_checkpoint.restore(model_path)
        print('Done.')

        print('Building data pipeline ...', end=' ')
        dataset = tf.data.TFRecordDataset(data_path)
        dataset = dataset.map(lambda xx: self._parse_train_(xx, self.mu_sig))
        dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        print('Done.')

        print('Training network ...')
        for epoch in range(self.epoch_shift+1, self.epoch_shift+self.N_epochs+1):
            print('Epoch: %d' % epoch)
            start_time = time()
            epoch_g_loss, epoch_d_loss, N = 0, 0, 0

            for batch in dataset:
                batch_idx, batch_LR, batch_HR = batch
                N_batch = batch_LR.shape[0]
                
                g_loss, d_loss, advers_perf = model.train_step(batch_LR, batch_HR)
                
                epoch_g_loss += g_loss.numpy() * N_batch
                epoch_d_loss += d_loss.numpy() * N_batch if d_loss is not None else 0
                N += N_batch

            epoch_g_loss = epoch_g_loss / N
            epoch_d_loss = epoch_d_loss / N if d_loss is not None else 0
            
            print('Epoch generator training loss=%.5f, discriminator training loss=%.5f' % (epoch_g_loss, epoch_d_loss))
            print('Epoch took %.2f seconds\n' % (time() - start_time), flush=True)

            if (epoch % self.save_every) == 0:
                g_saved_model = g_manager.save()
                gd_saved_model = gd_manager.save()
                print(f"Saved models at {g_saved_model} and {gd_saved_model}")

        g_saved_model = g_manager.save()
        gd_saved_model = gd_manager.save()
        print('Done.')
        return g_saved_model

    def test(self, r, data_path, model_path, batch_size=100, plot_data=False):
        assert self.mu_sig is not None, 'Value for mu_sig must be set first.'
        
        self.set_LR_data_shape(data_path)
        h, w, C = self.LR_data_shape

        print('Initializing network ...', end=' ')
        model = SR_NETWORK(r=r, C=C, status='testing')
        
        checkpoint = tf.train.Checkpoint(generator=model.generator)
        print('Loading saved network ...', end=' ')
        checkpoint.restore(model_path)
        print('Done.')

        print('Building data pipeline ...', end=' ')
        dataset = tf.data.TFRecordDataset(data_path)
        dataset = dataset.map(lambda xx: self._parse_test_(xx, self.mu_sig))
        dataset = dataset.batch(batch_size)
        print('Done.')

        print('Running test data ...')
        data_out = []
        idx_out = []
        
        for batch in dataset:
            batch_idx, batch_LR = batch
            batch_SR = model.generate(batch_LR, training=False)
            
            batch_LR = self.mu_sig[1] * batch_LR + self.mu_sig[0]
            batch_SR = self.mu_sig[1] * batch_SR + self.mu_sig[0]
            
            if plot_data:
                img_path = '/'.join([self.data_out_path, 'imgs'])
                if not os.path.exists(img_path):
                    os.makedirs(img_path)
                plot_SR_data(batch_idx.numpy(), batch_LR.numpy(), batch_SR.numpy(), img_path)
            
            data_out.append(batch_SR.numpy())
            idx_out.append(batch_idx.numpy())

        data_out = np.concatenate(data_out, axis=0)
        idx_out = np.concatenate(idx_out, axis=0)
        
        if not os.path.exists(self.data_out_path):
            os.makedirs(self.data_out_path)
        np.save(os.path.join(self.data_out_path, 'dataSR.npy'), data_out)
        
        print('Done.')

    @tf.autograph.experimental.do_not_convert
    def _parse_train_(self, example_proto, mu_sig=None):
        feature_description = {
            'index': tf.io.FixedLenFeature([], tf.int64),
            'data_LR': tf.io.FixedLenFeature([], tf.string),
            'h_LR': tf.io.FixedLenFeature([], tf.int64),
            'w_LR': tf.io.FixedLenFeature([], tf.int64),
            'data_HR': tf.io.FixedLenFeature([], tf.string),
            'h_HR': tf.io.FixedLenFeature([], tf.int64),
            'w_HR': tf.io.FixedLenFeature([], tf.int64),
            'c': tf.io.FixedLenFeature([], tf.int64)
        }
        example = tf.io.parse_single_example(example_proto, feature_description)

        idx = example['index']

        h_LR, w_LR = example['h_LR'], example['w_LR']
        h_HR, w_HR = example['h_HR'], example['w_HR']
        c = example['c']

        data_LR = tf.io.decode_raw(example['data_LR'], tf.float64)
        data_HR = tf.io.decode_raw(example['data_HR'], tf.float64)

        data_LR = tf.reshape(data_LR, (h_LR, w_LR, c))
        data_HR = tf.reshape(data_HR, (h_HR, w_HR, c))

        if mu_sig is not None:
            data_LR = (data_LR - mu_sig[0]) / mu_sig[1]
            data_HR = (data_HR - mu_sig[0]) / mu_sig[1]

        return idx, tf.cast(data_LR, tf.float32), tf.cast(data_HR, tf.float32)

    @tf.autograph.experimental.do_not_convert
    def _parse_test_(self, example_proto, mu_sig=None):
        feature_description = {
            'index': tf.io.FixedLenFeature([], tf.int64),
            'data_LR': tf.io.FixedLenFeature([], tf.string),
            'h_LR': tf.io.FixedLenFeature([], tf.int64),
            'w_LR': tf.io.FixedLenFeature([], tf.int64),
            'c': tf.io.FixedLenFeature([], tf.int64)
        }
        example = tf.io.parse_single_example(example_proto, feature_description)

        idx = example['index']
        h_LR, w_LR = example['h_LR'], example['w_LR']
        c = example['c']

        data_LR = tf.io.decode_raw(example['data_LR'], tf.float64)
        data_LR = tf.reshape(data_LR, (h_LR, w_LR, c))

        if mu_sig is not None:
            data_LR = (data_LR - mu_sig[0]) / mu_sig[1]

        return idx, tf.cast(data_LR, tf.float32)

    def set_mu_sig(self, data_path, batch_size=1):
        print('Loading data ...', end=' ')
        dataset = tf.data.TFRecordDataset(data_path)
        dataset = dataset.map(self._parse_train_)
        dataset = dataset.batch(batch_size)

        N, mu, sigma = 0, 0, 0
        for batch in dataset:
            _, _, data_HR = batch
            data_HR = data_HR.numpy()

            N_batch, h, w, c = data_HR.shape
            N_new = N + N_batch

            mu_batch = np.mean(data_HR, axis=(0, 1, 2))
            sigma_batch = np.var(data_HR, axis=(0, 1, 2))

            sigma = (N/N_new)*sigma + (N_batch/N_new)*sigma_batch + (N*N_batch/N_new**2)*(mu - mu_batch)**2
            mu = (N/N_new)*mu + (N_batch/N_new)*mu_batch

            N = N_new

        self.mu_sig = [mu, np.sqrt(sigma)]
        print('Done.')

    def set_LR_data_shape(self, data_path):
        print('Loading data ...', end=' ')
        dataset = tf.data.TFRecordDataset(data_path)
        dataset = dataset.map(self._parse_test_)
        dataset = dataset.batch(1)

        for batch in dataset:
            _, data_LR = batch
            self.LR_data_shape = data_LR.shape[1:]
            break
        print('Done. Final')