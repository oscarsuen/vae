import numpy as np
import matplotlib.pyplot as plt
from model import *

encoder.load_weights('encoder.h5')
decoder.load_weights('decoder.h5')

def _preprocess_label(sample):
    image = tf.cast(sample['image'], tf.float32) / 255.
    # image = image < tf.random.uniform(tf.shape(image))
    image = image < 0.5
    return image, sample['label']

label_eval = mnist['test'].map(_preprocess_label).batch(256).prefetch(tf.data.experimental.AUTOTUNE)

def display_imgs(x, y=None):
    if not isinstance(x, (np.ndarray, np.generic)):
        x = np.array(x)
    plt.ioff()
    n = x.shape[0]
    fig, axs = plt.subplots(1, n, figsize=(n, 1))
    if y is not None:
        fig.suptitle(np.argmax(y, axis=1))
    for i in range(n):
        axs.flat[i].imshow(x[i].squeeze(), interpolation='none', cmap='gray')
        axs.flat[i].axis('off')
    plt.show()
    plt.close()
    plt.ion()

# z = prior.sample(10)
# zhat = decoder(z)

# display_imgs(zhat.sample())
# display_imgs(zhat.mode())
# display_imgs(zhat.mean())

m = np.zeros((10, encoded_size), dtype=np.float32)
cnt = np.zeros((10, 1), dtype=np.float32)
for image, label in label_eval:
    m[label.numpy()] += encoder(image)
    cnt[label.numpy()] += 1
m = m/cnt

# mhat = decoder(m)

# display_imgs(mhat.sample())
# display_imgs(mhat.mode())
# display_imgs(mhat.mean())

for i in range(10):
    t = np.array([(1-a)*m[i%10]+(a)*m[(i+1)%10] for a in np.arange(0, 1.1, 0.1)])
    that = decoder(t)
    display_imgs(that.mode())
