import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class encoder(tf.keras.layers.Layer):
    def __init__(self, en_dim=64, featsize=1024, name="encoder"):
        super(encoder, self).__init__(name=name)
        self.en_dim = en_dim
        self.feat_size = featsize

        # sub layers
        self.conv1 = tf.keras.layers.Conv2D(filters=self.en_dim, kernel_size=5, strides=2, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=self.en_dim*2, kernel_size=5, strides=2, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=self.en_dim*4, kernel_size=5, strides=2, padding='same')
        self.conv4 = tf.keras.layers.Conv2D(filters=self.en_dim*8, kernel_size=5, strides=2, padding='same')
        self.dense1 = tf.keras.layers.Dense(units=self.feat_size)
        self.dense2 = tf.keras.layers.Dense(units=self.feat_size)
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        x = self.conv1(inputs)
        conv1 = self.lrelu(x)
        x = self.conv2(conv1)
        conv2 = self.lrelu(x)
        x = self.conv3(conv2)
        conv3 = self.lrelu(x)
        x = self.conv4(conv3)
        conv4 = self.lrelu(x)
        x = self.flatten(conv4)
        x = self.dense1(x)
        x = self.lrelu(x)
        x = self.dense2(x)
        x = self.lrelu(x)
        return x, [conv1, conv2, conv3, conv4]

class translator(tf.keras.layers.Layer):
    def __init__(self, featsize=1024, name="translator"):
        super(translator, self).__init__(name=name)
        self.featsize=featsize

        # sub layers
        self.dense1 = tf.keras.layers.Dense(units=self.featsize)
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, src_enc, tgt_enc):
        x = tf.keras.layers.concatenate([src_enc, tgt_enc])
        x = self.dense1(x)
        x = self.lrelu(x)
        return x

class decoder(tf.keras.layers.Layer):
    def __init__(self, de_dim=64, out_height=48, out_width=48, name="decoder"):
        super(decoder, self).__init__(name=name)
        self.de_dim = de_dim
        self.h = out_height
        self.w = out_width
        self.h16 = int(self.h/16)
        self.w16 = int(self.w/16)

        # sub layers
        self.dense1 = tf.keras.layers.Dense(units=self.de_dim*8*self.h16*self.w16)
        self.convtrans1 = tf.keras.layers.Conv2DTranspose(filters=self.de_dim*4, kernel_size=5, strides=2, padding='same')
        self.convtrans2 = tf.keras.layers.Conv2DTranspose(filters=self.de_dim*2, kernel_size=5, strides=2, padding='same')
        self.convtrans3 = tf.keras.layers.Conv2DTranspose(filters=self.de_dim, kernel_size=5, strides=2, padding='same')
        self.convtrans4 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=5, strides=2, padding='same')
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, input, features):
        x = self.dense1(input)
        x = self.lrelu(x)
        x = tf.reshape(x, [-1, self.h16, self.w16, self.de_dim*8])
        x = tf.keras.layers.concatenate([x, features[-1]])
        x = self.convtrans1(x)
        x = self.lrelu(x)
        x = tf.keras.layers.concatenate([x, features[-2]])
        x = self.convtrans2(x)
        x = self.lrelu(x)
        x = tf.keras.layers.concatenate([x, features[-3]])
        x = self.convtrans3(x)
        x = self.lrelu(x)
        x = tf.keras.layers.concatenate([x, features[-4]])
        x = self.convtrans4(x)
        return x

class IfO_model(tf.keras.Model):
    def __init__(self, en_dim=64, en_featsize=1024, tr_featsize=1024, de_dim=64, out_height=48, out_width=48
                 ,lambda1=1, lambda2=0.001, name='model'):
        super(IfO_model, self).__init__(name=name)
        self.en_dim = en_dim
        self.en_featsize = en_featsize
        self.tr_featsize = tr_featsize
        self.de_dim = de_dim
        self.out_height = out_height
        self.out_width = out_width
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.encoder_1 = encoder(self.en_dim, self.en_featsize, name="enccoder_1") # content encoder
        self.encoder_2 = encoder(self.en_dim, self.en_featsize, name="enccoder_2") # context encoder
        self.translator = translator(self.tr_featsize, name="translator")
        self.decoder = decoder(self.de_dim, self.out_height, self.out_width, name="decoder")

    def call(self, inputs):
        srcimg = inputs[0]
        trgctx = inputs[1]
        trgimg = inputs[2]

        z1, _ = self.encoder_1(srcimg)
        z2, ctx_features = self.encoder_2(trgctx)
        z3_hat = self.translator(z1, z2)
        z3, _ = self.encoder_1(trgimg)

        trans_img = self.decoder(z3_hat, ctx_features)
        recon_img = self.decoder(z3, ctx_features)

        self.loss_trans = tf.nn.l2_loss(trans_img-trgimg)
        self.loss_recon = tf.nn.l2_loss(recon_img-trgimg) * self.lambda1
        self.loss_align = tf.reduce_mean(tf.square(z3_hat-z3)) * self.lambda2
        self.loss = self.loss_trans + self.loss_recon + self.loss_align
        self.add_loss(self.loss)
        return trans_img, recon_img
