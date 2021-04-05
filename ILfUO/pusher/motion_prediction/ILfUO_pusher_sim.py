import tensorflow as tf
import numpy as np
import os
import logging
import imageio
import matplotlib.pyplot as plt
from datetime import datetime
from ILfUO_model import VideoGenerator

class ILfUO_pusher_sim():
    def __init__(self,
                 batch_size=100,
                 epochs = 10,
                 en_dim = 64,
                 de_dim = 64,
                 dc = 50,
                 dm = 10,
                 de = 20,
                 log_freq=100,
                 results_path='./results'):

        self.batch_size = batch_size
        self.epochs = epochs
        self.log_freq = log_freq
        self.results_path=results_path
        self.results_img_path = results_path + '/images'
        self.results_gif_path = results_path + '/gif'
        self.reuslts_log_path = results_path + "/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.en_dim = en_dim
        self.de_dim = de_dim
        self.dc = dc
        self.dm = dm
        self.de = de
        self.lambda_1 = 0.1
        self.lambda_2 = 5*1e-4

        if not os.path.exists(self.results_img_path):
            os.makedirs(self.results_img_path)
            os.makedirs(self.results_gif_path)

        # set logger
        self.mylogger = logging.getLogger("my")
        self.mylogger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.mylogger.addHandler(stream_handler)
        # tf.get_logger().setLevel('ERROR')

        # TensorBoard
        train_log_dir = self.reuslts_log_path + '/train'
        test_log_dir = self.reuslts_log_path + '/test'
        img_log_dir = self.reuslts_log_path + '/image'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        self.image_train_summary_writer = tf.summary.create_file_writer(img_log_dir + '/train')
        self.image_test_summary_writer = tf.summary.create_file_writer(img_log_dir + '/test')

        # data load
        self.load_data()

        # Model
        self.vid_gen = VideoGenerator(en_dim=self.en_dim, de_dim=self.de_dim, dc=self.dc, dm=self.dm)

        # optimizer
        self.ae_opt = tf.keras.optimizers.Adam(0.0001)
        # self.ae_opt = tf.keras.optimizers.RMSprop(0.0001)
        self.enc_opt = tf.keras.optimizers.Adam(0.00005)
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # define metrics
        self.train_ae_loss = tf.keras.metrics.Mean('train_ae_loss', dtype=tf.float32)
        self.train_recon_loss = tf.keras.metrics.Mean('train_recon_loss', dtype=tf.float32)
        self.train_enc_loss = tf.keras.metrics.Mean('train_enc_loss', dtype=tf.float32)
        self.train_z_c_loss = tf.keras.metrics.Mean('train_z_c_loss', dtype=tf.float32)
        self.train_z_m_loss = tf.keras.metrics.Mean('train_z_m_loss', dtype=tf.float32)

        self.test_ae_loss = tf.keras.metrics.Mean('test_ae_loss', dtype=tf.float32)
        self.test_recon_loss = tf.keras.metrics.Mean('test_recon_loss', dtype=tf.float32)
        self.test_enc_loss = tf.keras.metrics.Mean('test_enc_loss', dtype=tf.float32)
        self.test_z_c_loss = tf.keras.metrics.Mean('test_z_c_loss', dtype=tf.float32)
        self.test_z_m_loss = tf.keras.metrics.Mean('test_z_m_loss', dtype=tf.float32)

    def load_data(self):
        self.data = np.load("../../../dataset/singleview_push_sim_5000.npy")
        self.n_train = 4000
        self.n_valid = 1000
        self.traindata = np.swapaxes(self.data[:, :self.n_train], 0, 1)
        self.validdata = np.swapaxes(self.data[:, -self.n_valid:], 0, 1)
        vf_10 = np.ones(25,'int')*24
        vf_15 = np.ones(25, 'int') * 24
        vf_20 = np.ones(25, 'int') * 24
        vf_25 = np.ones(25, 'int') * 24
        vf_10[:10] = np.round(np.arange(1,11)*25/10).astype(int)-1
        vf_15[:15]= np.round(np.arange(1, 16) * 25 / 15).astype(int)-1
        vf_20[:20] = np.round(np.arange(1, 21) * 25 / 20).astype(int)-1
        vf_25[:25] = np.round(np.arange(1, 26) * 25 / 25).astype(int)-1
        self.traindata[:1000]=self.traindata[:1000,vf_10]
        self.traindata[1000:2000] = self.traindata[1000:2000, vf_15]
        self.traindata[2000:3000] = self.traindata[2000:3000, vf_20]
        self.traindata[3000:4000] = self.traindata[3000:4000, vf_25]
        self.validdata[:250] = self.validdata[:250,vf_10]
        self.validdata[250:500] = self.validdata[250:500, vf_15]
        self.validdata[500:750] = self.validdata[500:750, vf_20]
        self.validdata[750:1000] = self.validdata[750:1000, vf_25]
        # video length label
        self.trainvl = np.zeros([self.n_train, 24, 4], dtype='float32')
        self.trainvl[:1000, :, 0] = 1
        self.trainvl[1000:2000, :, 1] = 1
        self.trainvl[2000:3000, :, 2] = 1
        self.trainvl[3000:4000, :, 3] = 1
        self.validvl = np.zeros([self.n_train, 24, 4], dtype='float32')
        self.validvl[:250, :, 0] = 1
        self.validvl[250:500, :, 1] = 1
        self.validvl[500:750, :, 2] = 1
        self.validvl[750:1000, :, 3] = 1

    def train(self):
        tuples_id = np.arange(self.n_train)
        valid_id = np.arange(self.n_valid)
        bat_per_epo = int(self.n_train / self.batch_size)
        val_per_epo = int(self.n_valid / self.batch_size)
        self.mylogger.info("Batches (steps) per epoch: %d" % bat_per_epo)

        if tf.config.list_physical_devices('GPU'):
            device = '/device:GPU:0'
        else:
            device = '/device:CPU:0'
        with tf.device(device):
            for epoch in range(self.epochs):
                self.mylogger.info("\nStart of epoch : %d" % (epoch+1))
                np.random.shuffle(tuples_id)
                for step in range(bat_per_epo):
                    tupids = tuples_id[step*self.batch_size:(step+1)*self.batch_size]
                    vid_batch = self.traindata[tupids]
                    vl_batch = self.trainvl[tupids]
                    eps = tf.random.truncated_normal([self.batch_size, 24, self.de])
                    frames = np.swapaxes(np.array([np.arange(self.batch_size),
                                                   np.round(np.random.choice(25, self.batch_size, replace=True) / 25
                                                            * np.max(vl_batch[:, 0] * np.array([10, 15, 20, 25]),
                                                                     axis=1)).astype(int)]), 0, 1)

                    inputs = [vid_batch, vl_batch, eps, frames]

                    # Autoencoder update
                    with tf.GradientTape(persistent=True) as tape:
                        recon_vid, origin_vid, z_c, z_c_origin, z_m, z_m_origin, origin_frame, recon_frame = self.vid_gen(inputs)

                        recon_loss = tf.reduce_mean(tf.math.abs(origin_vid-recon_vid))
                        z_c_loss = tf.reduce_mean(tf.math.abs(z_c-z_c_origin))
                        z_m_loss = tf.reduce_mean(tf.math.abs(z_m-z_m_origin))
                        z_loss = self.lambda_2*(z_c_loss + z_m_loss)
                        enc_loss = tf.reduce_mean(tf.math.abs(origin_frame-recon_frame))

                        ae_loss = recon_loss + self.lambda_1*enc_loss

                    grads = tape.gradient(ae_loss, self.vid_gen.trainable_weights)
                    self.ae_opt.apply_gradients(zip(grads, self.vid_gen.trainable_weights))
                    grads = tape.gradient(z_loss, self.vid_gen.encoder_content.trainable_weights + self.vid_gen.encoder_motion.trainable_weights)
                    self.enc_opt.apply_gradients(zip(grads, self.vid_gen.encoder_content.trainable_weights + self.vid_gen.encoder_motion.trainable_weights))
                    self.train_ae_loss(ae_loss)
                    self.train_recon_loss(recon_loss)
                    self.train_enc_loss(enc_loss)
                    self.train_z_c_loss(z_c_loss)
                    self.train_z_m_loss(z_m_loss)

                    if (step+1) % self.log_freq == 0:
                        self.mylogger.info("epoch %d / %d, step %d / %d" % (epoch+1, self.epochs, step+1, bat_per_epo))
                        self.mylogger.info("\tae_loss = %.4f" % (ae_loss))
                        self.mylogger.info("\trecon_loss = %.4f" % (recon_loss))
                        self.mylogger.info("\tenc_loss = %.4f" % (enc_loss))

                with self.image_train_summary_writer.as_default():
                    images = tf.clip_by_value(tf.concat([(recon_vid[0]+1)/2., (origin_vid[0]+1)/2.], axis=2), 0, 1)
                    tf.summary.image("Train images", images, max_outputs=25, step=epoch)
                    images = tf.clip_by_value(tf.concat([(recon_frame[:10] + 1) / 2., (origin_frame[:10] + 1) / 2.], axis=2), 0, 1)
                    tf.summary.image("Train recover", images, max_outputs=25, step=epoch)

                with self.train_summary_writer.as_default():
                    tf.summary.scalar('ae_loss', self.train_ae_loss.result(), step=epoch)
                    tf.summary.scalar('recon_loss', self.train_recon_loss.result(), step=epoch)
                    tf.summary.scalar('enc_loss', self.train_enc_loss.result(), step=epoch)
                    tf.summary.scalar('z_c_loss', self.train_z_c_loss.result(), step=epoch)
                    tf.summary.scalar('z_m_loss', self.train_z_m_loss.result(), step=epoch)
                    tf.summary.histogram('z_m_origin', z_m_origin, step=epoch)
                    tf.summary.histogram('z_c_origin', z_c_origin, step=epoch)

                # Validation Step
                np.random.shuffle(valid_id)
                for step in range(val_per_epo):
                    tupids = valid_id[step * self.batch_size:(step + 1) * self.batch_size]
                    vid_batch = self.validdata[tupids]
                    vl_batch = self.validvl[tupids]
                    eps = tf.random.truncated_normal([self.batch_size, 24, self.de])
                    frames = np.swapaxes(np.array([np.arange(self.batch_size),
                                                   np.round(np.random.choice(25, self.batch_size, replace=True) / 25
                                                            * np.max(vl_batch[:, 0] * np.array([10, 15, 20, 25]),
                                                                     axis=1)).astype(int)]), 0, 1)

                    inputs = [vid_batch, vl_batch, eps, frames]

                    recon_vid, origin_vid, z_c, z_c_origin, z_m, z_m_origin, origin_frame, recon_frame = self.vid_gen(inputs, training=False)

                    recon_loss = tf.reduce_mean(tf.math.abs(origin_vid - recon_vid))
                    z_c_loss = tf.reduce_mean(tf.math.abs(z_c-z_c_origin))
                    z_m_loss = tf.reduce_mean(tf.math.abs(z_m-z_m_origin))
                    z_loss = self.lambda_2*(z_c_loss + z_m_loss)
                    enc_loss = tf.reduce_mean(tf.math.abs(origin_frame-recon_frame))
                    ae_loss = recon_loss + self.lambda_1*enc_loss

                    self.test_ae_loss(ae_loss)
                    self.test_recon_loss(recon_loss)
                    self.test_enc_loss(enc_loss)
                    self.test_z_c_loss(z_c_loss)
                    self.test_z_m_loss(z_m_loss)

                # saving imgae
                for i in range(25):
                    plt.subplot(5, 5, i + 1)
                    plt.imshow((recon_vid[0, i] + 1) / 2.)
                    plt.axis('off')
                # plt.show()
                plt.savefig(self.results_img_path + '/%d_recon' % (epoch + 1))
                plt.close()

                for i in range(25):
                    plt.subplot(5, 5, i + 1)
                    plt.imshow((origin_vid[0, i] + 1) / 2.)
                    plt.axis('off')
                # plt.show()
                plt.savefig(self.results_img_path + '/%d_src' % (epoch + 1))
                plt.close()

                with self.image_test_summary_writer.as_default():
                    images = tf.clip_by_value(tf.concat([(recon_vid[0]+1)/2., (origin_vid[0]+1)/2.], axis=2), 0, 1)
                    tf.summary.image("Validation images", images, max_outputs=25, step=epoch)
                    images = tf.clip_by_value(tf.concat([(recon_frame[:10] + 1) / 2., (origin_frame[:10] + 1) / 2.], axis=2), 0, 1)
                    tf.summary.image("Validation recover", images, max_outputs=25, step=epoch)

                with self.test_summary_writer.as_default():
                    tf.summary.scalar('ae_loss', self.test_ae_loss.result(), step=epoch)
                    tf.summary.scalar('recon_loss', self.test_recon_loss.result(), step=epoch)
                    tf.summary.scalar('enc_loss', self.test_enc_loss.result(), step=epoch)
                    tf.summary.scalar('z_c_loss', self.test_z_c_loss.result(), step=epoch)
                    tf.summary.scalar('z_m_loss', self.test_z_m_loss.result(), step=epoch)
                    tf.summary.histogram('z_m_origin', z_m_origin, step=epoch)
                    tf.summary.histogram('z_c_origin', z_c_origin, step=epoch)

                # reset metrics states
                self.train_recon_loss.reset_states()
                self.test_recon_loss.reset_states()

                if epoch == 0 or (epoch+1) % 10 == 0:
                    self.vid_gen.save(self.results_path + "/models/vid_gen_" + str(epoch + 1))
                    # saving gif
                    for i in range(10):
                        imageio.mimsave(self.results_gif_path + "/%d_recon_%d.gif" % (epoch + 1, i), recon_vid[i], duration=0.1)
                        imageio.mimsave(self.results_gif_path + "/%d_src_%d.gif" % (epoch + 1, i), origin_vid[i], duration=0.1)
                        plt.imshow((origin_vid[i, 0] + 1) / 2.)
                        plt.axis('off')
                        plt.savefig(self.results_gif_path + '/%d_src_%d' % (epoch + 1, i))


if __name__ == "__main__":
    ILfUO = ILfUO_pusher_sim(epochs=1000,
                             en_dim=64,
                             de_dim=64,
                             dc=200,
                             dm=40,
                             batch_size=40,
                             log_freq=50,
                             results_path='./results')
    ILfUO.train()