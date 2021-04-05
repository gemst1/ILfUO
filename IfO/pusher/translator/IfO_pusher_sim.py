import tensorflow as tf
import cv2
import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.io import gfile
from IfO_model import IfO_model
import sys
sys.path.append("../../../dataset")
from dataset import dataset_gen, transform, inverse_transform

class IfO():
    def __init__(self,
                 batch_size = 100,
                 epochs = 10,
                 seed = 1234,
                 dataset_path = "../../../dataset/singleview_push_sim_5000.npy",
                 results_path = "./results"):

        self.batch_size = batch_size
        self.epochs = epochs
        self.results_path = results_path

        # set random seed
        np.random.seed(seed)

        # Load Dataset
        if gfile.exists(dataset_path):
            self.vdata = np.load(dataset_path)
        else:
            self.vdata = dataset_gen()
        print("Data is loaded. Shape: ", self.vdata.shape)

        ntotal = self.vdata.shape[1]
        ntrain = 4500
        nvalid = ntotal - ntrain

        self.traindata = self.vdata[:, :ntrain]
        self.validdata = self.vdata[:, ntrain:]

        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path+"/images")
            os.makedirs(self.results_path + "/videos")

    def train(self):
        # Training

        # load model and optimizer
        self.model = IfO_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        loss_metric = tf.keras.metrics.Mean()

        ntuples_train = self.traindata.shape[0]*self.traindata.shape[1]*self.traindata.shape[1]

        tuples_id = np.arange(ntuples_train)
        print("Tuples per epoch: ", tuples_id.shape[0])
        bat_per_epo = int(tuples_id.shape[0]/self.batch_size)
        print("Batches (steps) per epoch: ", bat_per_epo)

        for epoch in range(self.epochs):
            print("Start of epoch ", epoch)
            np.random.shuffle(tuples_id)
            for step in range(bat_per_epo):
                # prepare data for training
                tupids = tuples_id[step*self.batch_size:(step+1)*self.batch_size]
                b, srcids = divmod(tupids, self.traindata.shape[1])
                trgfms, trgids = divmod(b, self.traindata.shape[1])
                srcdata = self.traindata[trgfms, srcids]
                trgdata = self.traindata[trgfms, trgids]
                trgctx = self.traindata[0, trgids]

                # training step
                with tf.GradientTape() as tape:
                    trans_img, recon_img = self.model([srcdata, trgctx, trgdata])
                    loss = self.model.losses
                grads = tape.gradient(loss, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                loss_metric(loss)

                if step % 200 == 0:
                    print("step %d / %d: mean_loss = %.4f" % (step, bat_per_epo, loss_metric.result()))

                    if step % 2000 == 0:
                        plt.imshow(inverse_transform(srcdata[0]))
                        plt.axis('off')
                        # plt.show()
                        plt.savefig('./results/images/%d_srcimg' % step)

                        plt.imshow(inverse_transform(trgctx[0]))
                        plt.axis('off')
                        # plt.show()
                        plt.savefig('./results/images/%d_trgctx' % step)

                        plt.imshow(inverse_transform(trgdata[0]))
                        plt.axis('off')
                        # plt.show()
                        plt.savefig('./results/images/%d_trgimg' % step)

                        plt.imshow(np.clip(inverse_transform(trans_img[0]), 0, 1))
                        plt.axis('off')
                        # plt.show()
                        plt.savefig('./results/images/%d_transimg' % step)

                        plt.imshow(np.clip(inverse_transform(recon_img[0]), 0, 1))
                        plt.axis('off')
                        # plt.show()
                        plt.savefig('./results/images/%d_reconimg' % step)

                        self.model.save('./results/models/model_%d' % step)

                if loss_metric.result() < 2300 or step==80000:
                    self.model.save('./results/models/model_%d' % step)
                    if step >= 80000:
                        break

    def evaluation(self, restore_path=None, eval_eps = 10, save=True):
        if restore_path:
            self.model = tf.keras.models.load_model(restore_path)

        # select evaluation episodes
        eval_data = self.validdata[:, np.random.choice(self.validdata.shape[1], eval_eps*2)]
        src_data = np.reshape(eval_data[:, :eval_eps], (-1, 48, 48, 3), 'F')
        trg_data = eval_data[:, eval_eps:]
        trg_ctx = np.reshape(np.array([trg_data[0] for _ in range(25)]), (-1, 48, 48, 3), 'F')
        trg_data = np.reshape(trg_data, (-1, 48, 48, 3), 'F')

        # translation
        trans_imgs, _ = self.model([src_data, trg_ctx, trg_data])

        # save episodes as mp4 video and gif
        if save:
            trans_imgs = (np.clip(tf.image.resize(inverse_transform(trans_imgs), [480, 480], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), 0, 1)*255).astype(np.uint8)
            src_imgs = (np.clip(tf.image.resize(inverse_transform(src_data), [480, 480], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), 0, 1) * 255).astype(np.uint8)
            trg_imgs = (np.clip(tf.image.resize(inverse_transform(trg_data), [480, 480], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), 0, 1) * 255).astype(np.uint8)

            for i in range(eval_eps):
                out_trans = cv2.VideoWriter('./results/videos/episode_%d_trans.mp4' % i, cv2.VideoWriter_fourcc(*'MP4V'), 10, (480, 480))
                out_src = cv2.VideoWriter('./results/videos/episode_%d_src.mp4' % i, cv2.VideoWriter_fourcc(*'MP4V'), 10, (480, 480))
                out_trg = cv2.VideoWriter('./results/videos/episode_%d_trg.mp4' % i, cv2.VideoWriter_fourcc(*'MP4V'), 10, (480, 480))
                for j in range(25):
                    # mp4 video
                    out_trans.write(cv2.cvtColor(trans_imgs[i * 25 + j], cv2.COLOR_RGB2BGR))
                    out_src.write(cv2.cvtColor(src_imgs[i * 25 + j], cv2.COLOR_RGB2BGR))
                    out_trg.write(cv2.cvtColor(trg_imgs[i * 25 + j], cv2.COLOR_RGB2BGR))
                out_trans.release()
                out_src.release()
                out_trg.release()
                # gif
                imageio.mimsave("./results/videos/episode_%d_trans.gif" % i, trans_imgs[i * 25:(i + 1) * 25], duration=0.1)
                imageio.mimsave("./results/videos/episode_%d_src.gif" % i, src_imgs[i * 25:(i + 1) * 25], duration=0.1)
                imageio.mimsave("./results/videos/episode_%d_trg.gif" % i, trg_imgs[i * 25:(i + 1) * 25], duration=0.1)

if __name__ == "__main__":
    IfO_pusher = IfO(batch_size=500,
                     epochs=1)
    IfO_pusher.train()
    IfO_pusher.evaluation('./results/models/model_80000', 20)