from model import *

vae.summary()
vae.fit(data_train, epochs=15, validation_data=data_eval)
encoder.save_weights('encoder.h5')
decoder.save_weights('decoder.h5')
vae.save_weights('vae.h5')
