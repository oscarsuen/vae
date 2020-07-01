import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
tfk = tf.keras
tfkl = tfk.layers
tfpl = tfp.layers
tfd = tfp.distributions

tf.enable_eager_execution()

mnist, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=False)
def _preprocess(sample):
    image = tf.cast(sample['image'], tf.float32) / 255.
    # image = image < tf.random.uniform(tf.shape(image))
    image = image < 0.5
    return image, image
data_train = mnist['train'] \
                .map(_preprocess) \
                .batch(256) \
                .prefetch(tf.data.experimental.AUTOTUNE) \
                .shuffle(int(1e4))
data_eval = mnist['test'] \
                .map(_preprocess) \
                .batch(256) \
                .prefetch(tf.data.experimental.AUTOTUNE)

input_shape = mnist_info.features['image'].shape
encoded_size = 16
base_depth = 32

prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1), reinterpreted_batch_ndims=1)

encoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=input_shape),
    tfkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
    tfkl.Conv2D(1*base_depth, 5, strides=1, padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2D(1*base_depth, 5, strides=2, padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2D(2*base_depth, 5, strides=1, padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2D(2*base_depth, 5, strides=2, padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2D(4*base_depth, 7, strides=1, padding='valid', activation=tf.nn.leaky_relu),
    tfkl.Flatten(),
    tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size), activation=None),
    tfpl.MultivariateNormalTriL(encoded_size, activity_regularizer=tfpl.KLDivergenceRegularizer(prior)),
])

decoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=[encoded_size]),
    tfkl.Reshape([1, 1, encoded_size]),
    tfkl.Conv2DTranspose(2*base_depth, 7, strides=1, padding='valid', activation=tf.nn.leaky_relu),
    tfkl.Conv2DTranspose(2*base_depth, 5, strides=1, padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2DTranspose(2*base_depth, 5, strides=2, padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2DTranspose(1*base_depth, 5, strides=1, padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2DTranspose(1*base_depth, 5, strides=2, padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2DTranspose(1*base_depth, 5, strides=1, padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2D(filters=1, kernel_size=5, strides=1, padding='same', activation=None),
    tfkl.Flatten(),
    tfpl.IndependentBernoulli(input_shape, tfd.Bernoulli.logits),
])

vae = tfk.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs[0]))
vae.compile(optimizer=tfk.optimizers.Adam(learning_rate=1e-3), loss=lambda x, rv_x: -rv_x.log_prob(x))
