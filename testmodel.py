import tensorflow as tf
model = tf.saved_model.load('my_saved_model')
print(list(model.signatures.keys()))