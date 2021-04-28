import tensorflow as tf

class encoder(tf.keras.layers.Layer):
    def __init__(self, en_dim=16, feat_size=128, bn=False, name="encoder"):
        super(encoder, self).__init__(name=name)
        self.en_dim = en_dim
        self.feat_size = feat_size
        self.bn = bn

        # sub layers
        self.conv1 = tf.keras.layers.Conv2D(filters=self.en_dim, kernel_size=3, strides=2, padding='same', use_bias=False) # 48 * 48 *3 -> 24 * 24 * (en_dim)
        self.conv2 = tf.keras.layers.Conv2D(filters=self.en_dim * 2, kernel_size=3, strides=2, padding='same', use_bias=False) # -> 12 * 12 * (en_dim * 2)
        self.conv3 = tf.keras.layers.Conv2D(filters=self.en_dim * 4, kernel_size=3, strides=2, padding='same', use_bias=False) # -> 6 * 6 * (en_dim * 4)
        self.conv4 = tf.keras.layers.Conv2D(filters=self.en_dim * 8, kernel_size=3, strides=2, padding='same', use_bias=False) # -> 3 * 3 * (en_dim * 8)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(self.feat_size)
        self.dense2 = tf.keras.layers.Dense(self.feat_size)
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)

        # batch nomalization
        if self.bn:
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.bn3 = tf.keras.layers.BatchNormalization()
            self.bn4 = tf.keras.layers.BatchNormalization()
            self.bn5 = tf.keras.layers.BatchNormalization()

    def call(self, x, training=True):
        x = self.conv1(x)
        conv1 = self.lrelu(x)
        x = self.conv2(conv1)
        if self.bn:
            x = self.bn1(x, training)
        conv2 = self.lrelu(x)
        x = self.conv3(conv2)
        if self.bn:
            x = self.bn2(x, training)
        conv3 = self.lrelu(x)
        x = self.conv4(conv3)
        if self.bn:
            x = self.bn3(x, training)
        conv4 = self.lrelu(x)
        x = self.flatten(conv4)
        x = self.dense1(x)
        if self.bn:
            x = self.bn4(x, training)
        x = self.lrelu(x)
        x = self.dense2(x)
        if self.bn:
            x = self.bn5(x, training)
        x = self.lrelu(x)

        return x, [conv1, conv2, conv3, conv4]

class Rm(tf.keras.layers.Layer):
    def __init__(self, feat_size, unroll=False, name="Rm"):
        super(Rm, self).__init__(name=name)
        self.feat_size = feat_size

        self.gru1 = tf.keras.layers.GRU(self.feat_size, return_sequences=True, unroll=unroll)

    def call(self, x, initial_state=None):
        x = self.gru1(x, initial_state=initial_state)
        return x

class decoder(tf.keras.layers.Layer):
    def __init__(self, de_dim=16, out_height=48, out_width=48, bn=False, name="decoder"):
        super(decoder, self).__init__(name=name)
        self.h = out_height
        self.w = out_width
        self.de_dim = de_dim
        self.h16 = int(self.h / 16)
        self.w16 = int(self.w / 16)
        self.bn = bn

        # sub layers
        self.dense1 = tf.keras.layers.Dense(self.de_dim * 8 * self.h16 * self.w16) # -> 3 * 3 * (de_dim*8)
        self.convtrans1 = tf.keras.layers.Conv2DTranspose(filters=self.de_dim * 4, kernel_size=3, strides=2, padding='same', use_bias=False) # -> 6 * 6 * (de_dim*4)
        self.convtrans2 = tf.keras.layers.Conv2DTranspose(filters=self.de_dim * 2, kernel_size=3, strides=2, padding='same', use_bias=False)  # -> 12 * 12 * (de_dim*2)
        self.convtrans3 = tf.keras.layers.Conv2DTranspose(filters=self.de_dim, kernel_size=3, strides=2, padding='same', use_bias=False)  # -> 24 * 24 * (de_dim)
        self.convtrans4 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=2, padding='same', use_bias=False)  # -> 48 * 48 * 3
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.tanh = tf.keras.layers.Activation('tanh')

        # batch nomalization
        if self.bn:
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.bn3 = tf.keras.layers.BatchNormalization()

    def __call__(self, inputs, training=True):
        x, f0, f1, f2, f3 = inputs

        x = self.dense1(x)
        x = self.lrelu(x)
        x = tf.reshape(x, [-1, self.h16, self.w16, self.de_dim * 8])
        x = tf.keras.layers.concatenate([x, f3])
        x = self.convtrans1(x)
        if self.bn:
            x = self.bn1(x, training)
        x = self.lrelu(x)
        x = tf.keras.layers.concatenate([x, f2])
        x = self.convtrans2(x)
        if self.bn:
            x = self.bn2(x, training)
        x = self.lrelu(x)
        x = tf.keras.layers.concatenate([x, f1])
        x = self.convtrans3(x)
        if self.bn:
            x = self.bn3(x, training)
        x = self.lrelu(x)
        x = tf.keras.layers.concatenate([x, f0])
        x = self.convtrans4(x)
        x = self.tanh(x)

        return x

class dist_decoder(tf.keras.layers.Layer):
    def __init__(self, de_dim=16, out_height=48, out_width=48, bn=False, name="dist_decoder"):
        super(dist_decoder, self).__init__(name=name)
        self.h = out_height
        self.w = out_width
        self.de_dim = de_dim
        self.h16 = int(self.h / 16)
        self.w16 = int(self.w / 16)
        self.bn = bn

        # sub layers
        self.dense1 = tf.keras.layers.Dense(self.de_dim * 8 * self.h16 * self.w16) # -> 3 * 3 * (de_dim*8)
        self.convtrans1 = tf.keras.layers.Conv2DTranspose(filters=self.de_dim * 4, kernel_size=3, strides=2, padding='same', use_bias=False) # -> 6 * 6 * (de_dim*4)
        self.convtrans2 = tf.keras.layers.Conv2DTranspose(filters=self.de_dim * 2, kernel_size=3, strides=2, padding='same', use_bias=False)  # -> 12 * 12 * (de_dim*2)
        self.convtrans3 = tf.keras.layers.Conv2DTranspose(filters=self.de_dim, kernel_size=3, strides=2, padding='same', use_bias=False)  # -> 24 * 24 * (de_dim)
        self.convtrans4 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=2, padding='same', use_bias=False)  # -> 48 * 48 * 3
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.tanh = tf.keras.layers.Activation('tanh')

        # batch nomalization
        if self.bn:
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.bn3 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=True):
        z, f0, f1, f2, f3 = inputs
        vid_len = z.shape[1]

        x = tf.keras.layers.TimeDistributed(self.dense1)(z, training)
        x = self.lrelu(x)
        x = tf.reshape(x, [-1, vid_len, self.h16, self.w16, self.de_dim * 8])
        x = tf.concat([x, f3], axis=-1)
        x = tf.keras.layers.TimeDistributed(self.convtrans1)(x, training)
        if self.bn:
            x = tf.keras.layers.TimeDistributed(self.bn1)(x, training)
        x = self.lrelu(x)
        x = tf.concat([x, f2], axis=-1)
        x = tf.keras.layers.TimeDistributed(self.convtrans2)(x, training)
        if self.bn:
            x = tf.keras.layers.TimeDistributed(self.bn2)(x, training)
        x = self.lrelu(x)
        x = tf.concat([x, f1], axis=-1)
        x = tf.keras.layers.TimeDistributed(self.convtrans3)(x, training)
        if self.bn:
            x = tf.keras.layers.TimeDistributed(self.bn3)(x, training)
        x = self.lrelu(x)
        x = tf.concat([x, f0], axis=-1)
        x = tf.keras.layers.TimeDistributed(self.convtrans4)(x, training)
        x = self.tanh(x)

        return x

class VideoGenerator(tf.keras.Model):
    def __init__(self, en_dim=128, de_dim=128, dc=1000, dm=200, de=20, name="vid_gen"):
        super(VideoGenerator, self).__init__(name=name)
        self.en_dim = en_dim
        self.de_dim = de_dim
        self.dc = dc
        self.dm = dm
        self.de = de

        # sub layers
        self.Rm = Rm(feat_size=self.dm, unroll=False, name='rm')
        self.encoder_content = encoder(en_dim=self.en_dim, feat_size=self.dc, bn=True, name='enc_con')
        self.encoder_motion = encoder(en_dim=self.en_dim, feat_size=self.dm, bn=True, name='enc_mot')
        self.decoder = dist_decoder(de_dim=self.de_dim, bn=True, name='dec')

        # decoder
        self.h = 48
        self.w = 48
        self.de_dim = de_dim
        self.h16 = int(self.h / 16)
        self.w16 = int(self.w / 16)

    def call(self, inputs, training=True):
        vid_batch, vl_batch, eps, frames = inputs
        batch_size = vid_batch.shape[0]

        z_c, features = self.encoder_content(vid_batch[:, 0], training)
        z_m, _ = self.encoder_motion(vid_batch[:, 0], training)
        # # sample noise
        # # eps = tf.random.truncated_normal([batch_size, 24, self.de])
        eps = tf.concat([eps, vl_batch], axis=-1)
        # # generate state sequence
        f0 = tf.tile(tf.reshape(features[0], [-1, 1, 24, 24, self.en_dim]), [1, 25, 1, 1, 1])
        f1 = tf.tile(tf.reshape(features[1], [-1, 1, 12, 12, self.en_dim*2]), [1, 25, 1, 1, 1])
        f2 = tf.tile(tf.reshape(features[2], [-1, 1, 6, 6, self.en_dim*4]), [1, 25, 1, 1, 1])
        f3 = tf.tile(tf.reshape(features[3], [-1, 1, 3, 3, self.en_dim*8]), [1, 25, 1, 1, 1])
        z_cs = tf.reshape(tf.tile(z_c, [1, 25]), [-1, 25, self.dc])
        z_ms = tf.concat([tf.reshape(z_m, [-1,1,self.dm]), self.Rm(eps, initial_state=z_m)], axis=1)
        z = tf.concat([z_cs, z_ms], axis=-1)

        # decoder
        inputs_dec = [z, f0, f1, f2, f3]
        x = self.decoder(inputs_dec, training)

        recon_vid = x
        origin_vid = vid_batch

        origin_frame = tf.gather_nd(origin_vid, frames)
        z_c_origin, _ = self.encoder_content(origin_frame, training)
        z_m_origin, _ = self.encoder_motion(origin_frame, training)
        z_m = tf.gather_nd(z_ms, frames)
        # reconstruction selected original video frames
        f0 = tf.reshape(features[0], [-1, 1, 24, 24, self.en_dim])
        f1 = tf.reshape(features[1], [-1, 1, 12, 12, self.en_dim * 2])
        f2 = tf.reshape(features[2], [-1, 1, 6, 6, self.en_dim * 4])
        f3 = tf.reshape(features[3], [-1, 1, 3, 3, self.en_dim * 8])
        z_cs = tf.reshape(z_c, [-1, 1, self.dc])
        z_m_origins = tf.reshape(z_m_origin, [-1, 1, self.dm])
        z = tf.concat([z_cs, z_m_origins], axis=-1)

        # decoder
        inputs_dec = [z, f0, f1, f2, f3]
        recon_frame = self.decoder(inputs_dec, training)[:,0]

        return recon_vid, origin_vid, z_c, z_c_origin, z_m, z_m_origin, origin_frame, recon_frame
